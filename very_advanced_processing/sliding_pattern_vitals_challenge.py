from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "scipy>=1.13",
#   "matplotlib>=3.8",
# ]
# ///

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal


@dataclass(slots=True)
class PatternStageConfig:
    name: str
    frequency_band_hz: tuple[float, float]
    window_s: float
    candidate_count: int = 10
    max_templates_to_average: int = 5
    min_events: int = 3
    min_corr_peak: float = 0.15
    peak_prominence: float = 0.04
    peak_distance_factor: float = 0.72
    fit_margin_s: float = 0.6
    stretch_bounds: tuple[float, float] = (0.65, 1.45)
    max_fit_events: int | None = None
    subtraction_taper_fraction: float = 0.06
    highpass_cutoff_hz: float | None = None


@dataclass(slots=True)
class ExtractionConfig:
    stages: tuple[PatternStageConfig, ...] = (
        PatternStageConfig(
            name="breath",
            frequency_band_hz=(0.18, 0.60),
            window_s=4.0,
            candidate_count=12,
            max_templates_to_average=5,
            min_events=5,
            min_corr_peak=0.18,
            peak_prominence=0.05,
            fit_margin_s=0.9,
            stretch_bounds=(0.62, 1.55),
            subtraction_taper_fraction=0.08,
        ),
        PatternStageConfig(
            name="heart",
            frequency_band_hz=(0.75, 3.0),
            window_s=1.05,
            candidate_count=14,
            max_templates_to_average=6,
            min_events=8,
            min_corr_peak=0.12,
            peak_prominence=0.025,
            fit_margin_s=0.28,
            stretch_bounds=(0.62, 1.45),
            subtraction_taper_fraction=0.08,
            highpass_cutoff_hz=0.55,
        ),
    )
    smooth_frequency_s: float = 2.0
    smooth_amplitude_s: float = 2.0


@dataclass(slots=True)
class PatternFit:
    start_index_guess: int
    start_time_s: float
    duration_s: float
    tau_s: float
    stretch: float
    beta0: float
    beta1: float
    offset: float
    amplitude: float
    cost: float
    success: bool


@dataclass(slots=True)
class CandidatePattern:
    template_start_index: int
    template: np.ndarray
    correlation: np.ndarray
    correlation_time_s: np.ndarray
    peaks: np.ndarray
    peak_heights: np.ndarray
    periodicity_hz: float
    periodicity_cv: float
    score: float
    average_pattern: np.ndarray | None = None
    fits: list[PatternFit] = field(default_factory=list)


@dataclass(slots=True)
class PatternStageResult:
    name: str
    input_signal: np.ndarray
    component: np.ndarray
    residual_signal: np.ndarray
    pattern: np.ndarray
    pattern_time_s: np.ndarray
    event_times_s: np.ndarray
    event_start_indices: np.ndarray
    instantaneous_frequency_hz: np.ndarray
    instantaneous_amplitude: np.ndarray
    correlation: np.ndarray
    correlation_time_s: np.ndarray
    fits: list[PatternFit]
    candidates: list[CandidatePattern]


@dataclass(slots=True)
class VitalPatternExtractionResult:
    fs: float
    time_s: np.ndarray
    stages: list[PatternStageResult]
    residual_signal: np.ndarray

    def by_name(self, name: str) -> PatternStageResult:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)


def extract_vital_patterns(
    sig: np.ndarray,
    fs: float,
    config: ExtractionConfig | None = None,
) -> VitalPatternExtractionResult:
    """Extract repeated respiration-like and heart-like patterns from displacement.

    The method follows the requested deterministic workflow:
    candidate windows are correlated with the signal, periodic candidates are
    fitted with a time-stretch and affine-amplitude least-squares model, fitted
    occurrences are averaged into a cleaner pattern, the pattern is fitted back
    onto the signal, and the fitted component is subtracted before the next
    faster stage is extracted.
    """

    cfg = config or ExtractionConfig()
    y = _as_float_1d(sig)
    t = np.arange(y.size, dtype=float) / fs
    residual = y.copy()
    stages: list[PatternStageResult] = []

    for stage_cfg in cfg.stages:
        stage = _extract_one_stage(residual, fs, stage_cfg, cfg)
        stages.append(stage)
        residual = stage.residual_signal

    return VitalPatternExtractionResult(
        fs=float(fs),
        time_s=t,
        stages=stages,
        residual_signal=residual,
    )


def introductory_pattern_extraction_demo(
    sig: np.ndarray,
    fs: float,
    config: ExtractionConfig | None = None,
    true_frequency_by_stage: dict[str, np.ndarray] | None = None,
    true_amplitude_by_stage: dict[str, np.ndarray | float] | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> tuple[VitalPatternExtractionResult, list[object]]:
    """Run the extraction and create compact explanatory matplotlib figures."""

    import matplotlib.pyplot as plt

    result = extract_vital_patterns(sig, fs, config)
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    figures: list[object] = []
    t = result.time_s

    fig, axes = plt.subplots(len(result.stages) + 1, 1, sharex=True, figsize=(12, 7))
    axes[0].plot(t, sig, lw=1.1, label="input")
    axes[0].set_title("Input signal and extracted components")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    for ax, stage in zip(axes[1:], result.stages, strict=True):
        ax.plot(t, stage.input_signal, lw=0.8, alpha=0.45, label=f"{stage.name} stage input")
        ax.plot(t, stage.component, lw=1.2, label=f"fitted {stage.name}")
        ax.plot(t, stage.residual_signal, lw=0.9, label="residual")
        ax.set_ylabel("disp.")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    _save_fig(fig, output_path, "00_components.png")
    figures.append(fig)

    for stage in result.stages:
        figures.extend(_plot_stage_diagnostics(stage, result.time_s, fs, output_path))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    for stage in result.stages:
        axes[0].plot(t, stage.instantaneous_frequency_hz, label=f"{stage.name} estimate")
        axes[1].plot(t, stage.instantaneous_amplitude, label=f"{stage.name} estimate")
        if true_frequency_by_stage and stage.name in true_frequency_by_stage:
            axes[0].plot(t, true_frequency_by_stage[stage.name], "--", label=f"{stage.name} truth")
        if true_amplitude_by_stage and stage.name in true_amplitude_by_stage:
            amp_truth = true_amplitude_by_stage[stage.name]
            if np.isscalar(amp_truth):
                axes[1].axhline(float(amp_truth), ls="--", label=f"{stage.name} truth")
            else:
                axes[1].plot(t, amp_truth, "--", label=f"{stage.name} truth")
    axes[0].set_ylabel("frequency [Hz]")
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("time [s]")
    axes[0].set_title("Instantaneous metrics")
    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path, "90_instantaneous_metrics.png")
    figures.append(fig)

    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

    return result, figures


def benchmark_challenge_signal(
    fs: float = 20.0,
    duration_s: float = 60.0,
    heart_scale: float = 0.1,
    noise_std: float = 0.0,
    br_mod_scale: float = 0.15,
    hr_mod_scale: float = 0.20,
    seed: int = 1234,
) -> dict[str, np.ndarray | float]:
    """Generate the synthetic signal from the challenge with tunable slopes/noise."""

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration_s, int(round(fs * duration_s)), endpoint=False)
    br_f = 0.3 + trapezoid_smooth(t / 20.0, fs, init_phase=0.75, lowpass_freq=0.2) * br_mod_scale - 0.05
    hr_f = 1.0 + trapezoid_smooth(t / 40.0, fs, init_phase=0.75, lowpass_freq=0.1) * hr_mod_scale - 0.05

    br_phase = 2.0 * np.pi * np.cumsum(br_f) / fs
    breath = (
        3.66 * np.sin(br_phase + 2.56)
        + 0.91 * np.sin(2.0 * br_phase + 2.17)
        + 0.22 * np.sin(3.0 * br_phase - 0.51)
        + 0.23 * np.sin(4.0 * br_phase - 1.1)
    ) / 3.66

    hr_phase_cycles = np.cumsum(hr_f) / fs
    heart_unit = ekg_like(hr_phase_cycles % 1.0) / 0.65
    heart = heart_scale * heart_unit
    sig = breath + heart
    if noise_std > 0.0:
        sig = sig + rng.normal(0.0, noise_std, size=sig.size)

    return {
        "fs": fs,
        "time_s": t,
        "signal": sig,
        "breath": breath,
        "heart": heart,
        "breath_frequency_hz": br_f,
        "heart_frequency_hz": hr_f,
        "breath_amplitude": robust_half_peak_to_peak(breath),
        "heart_amplitude": robust_half_peak_to_peak(heart),
    }


def score_extraction_against_truth(
    result: VitalPatternExtractionResult,
    truth: dict[str, np.ndarray | float],
    grace_s: float = 10.0,
) -> dict[str, float]:
    """Compute simple post-grace MAE/RMSE metrics for frequency and amplitude."""

    t = result.time_s
    mask = t >= grace_s
    metrics: dict[str, float] = {}
    truth_map = {
        "breath": ("breath_frequency_hz", "breath_amplitude"),
        "heart": ("heart_frequency_hz", "heart_amplitude"),
    }
    for stage in result.stages:
        freq_key, amp_key = truth_map.get(stage.name, (None, None))
        if freq_key and freq_key in truth:
            est = stage.instantaneous_frequency_hz
            valid = mask & np.isfinite(est)
            if np.any(valid):
                err = est[valid] - np.asarray(truth[freq_key])[valid]
                metrics[f"{stage.name}_freq_mae_hz"] = float(np.mean(np.abs(err)))
                metrics[f"{stage.name}_freq_rmse_hz"] = float(np.sqrt(np.mean(err**2)))
        if amp_key and amp_key in truth:
            est_amp = stage.instantaneous_amplitude
            valid = mask & np.isfinite(est_amp)
            if np.any(valid):
                amp_truth = truth[amp_key]
                if np.isscalar(amp_truth):
                    ref_amp = np.full(np.count_nonzero(valid), float(amp_truth))
                else:
                    ref_amp = np.asarray(amp_truth)[valid]
                err_amp = est_amp[valid] - ref_amp
                metrics[f"{stage.name}_amp_mae"] = float(np.mean(np.abs(err_amp)))
                metrics[f"{stage.name}_amp_rmse"] = float(np.sqrt(np.mean(err_amp**2)))
    return metrics


def ekg_like(x: np.ndarray) -> np.ndarray:
    return np.exp(-50.0 * (x - 0.2) ** 2) - 0.5 * np.exp(-80.0 * (x - 0.25) ** 2)


def trapezoid_smooth(
    phase: np.ndarray,
    fs: float,
    flat_duty: float = 0.5,
    init_phase: float = 0.0,
    lowpass_freq: float = 0.1,
) -> np.ndarray:
    phase = np.asarray(phase)
    single_ramp = (1.0 - flat_duty) / 2.0
    single_flat = flat_duty / 2.0
    r1_start = 0.0
    r1_end = single_ramp
    r2_start = single_flat + single_ramp
    r2_end = single_flat + 2.0 * single_ramp
    fi = (phase - init_phase) % 1.0
    out = np.zeros_like(fi, dtype=float)
    ramp_up = (r1_start < fi) & (fi < r1_end)
    high = (r1_end <= fi) & (fi <= r2_start)
    ramp_down = (r2_start < fi) & (fi < r2_end)
    out[ramp_up] = (fi[ramp_up] - r1_start) / (r1_end - r1_start)
    out[high] = 1.0
    out[ramp_down] = (r2_end - fi[ramp_down]) / (r2_end - r2_start)
    b, a = scipy.signal.butter(2, lowpass_freq, btype="lowpass", fs=fs)
    return scipy.signal.filtfilt(b, a, out)


def robust_half_peak_to_peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(0.5 * (np.nanpercentile(x, 95.0) - np.nanpercentile(x, 5.0)))


def _extract_one_stage(
    y: np.ndarray,
    fs: float,
    stage_cfg: PatternStageConfig,
    global_cfg: ExtractionConfig,
) -> PatternStageResult:
    n = y.size
    m = max(5, int(round(stage_cfg.window_s * fs)))
    y_detect = _stage_detection_signal(y, fs, stage_cfg)
    starts = _candidate_starts(n, m, stage_cfg.candidate_count, fs)
    candidates: list[CandidatePattern] = []

    for start in starts:
        template = _normalize_pattern(y_detect[start : start + m])
        corr, corr_t = normalized_sliding_correlation(y_detect, template, fs)
        peaks, heights = _find_correlation_peaks(corr, fs, stage_cfg)
        freq, cv = _periodicity_from_peaks(peaks, fs)
        score = _candidate_score(freq, cv, heights, len(peaks), stage_cfg)
        candidates.append(
            CandidatePattern(
                template_start_index=int(start),
                template=template,
                correlation=corr,
                correlation_time_s=corr_t,
                peaks=peaks,
                peak_heights=heights,
                periodicity_hz=freq,
                periodicity_cv=cv,
                score=score,
            )
        )

    selected = _select_candidates(candidates, stage_cfg)
    if not selected:
        selected = sorted(candidates, key=lambda c: c.score, reverse=True)[:1]

    average_patterns = []
    for candidate in selected:
        fit_peaks = _event_subset(candidate.peaks, stage_cfg.max_fit_events)
        candidate.fits = [
            _fit_pattern_at_start(y_detect, candidate.template, int(peak), fs, stage_cfg)
            for peak in fit_peaks
        ]
        candidate.average_pattern = _average_fitted_patterns(
            y_detect,
            candidate.template,
            candidate.fits,
            fs,
            m,
        )
        if candidate.average_pattern is not None:
            average_patterns.append(candidate.average_pattern)

    if average_patterns:
        pattern = _normalize_pattern(np.nanmedian(np.vstack(average_patterns), axis=0))
    else:
        pattern = selected[0].template

    # Re-detect and re-fit with the consensus pattern. This is the pattern used
    # for subtraction and instantaneous metrics.
    corr, corr_t = normalized_sliding_correlation(y_detect, pattern, fs)
    peaks, _ = _find_correlation_peaks(corr, fs, stage_cfg)
    fits = [
        _fit_pattern_at_start(y_detect, pattern, int(peak), fs, stage_cfg)
        for peak in _event_subset(peaks, stage_cfg.max_fit_events)
    ]
    refined = _average_fitted_patterns(y_detect, pattern, fits, fs, m)
    if refined is not None:
        pattern = refined
        corr, corr_t = normalized_sliding_correlation(y_detect, pattern, fs)
        peaks, _ = _find_correlation_peaks(corr, fs, stage_cfg)
        fits = [
            _fit_pattern_at_start(y_detect, pattern, int(peak), fs, stage_cfg)
            for peak in _event_subset(peaks, stage_cfg.max_fit_events)
        ]

    fiducial_event_times: np.ndarray | None = None
    fiducial_event_indices: np.ndarray | None = None
    if stage_cfg.highpass_cutoff_hz is not None:
        expected_freq = selected[0].periodicity_hz if selected and np.isfinite(selected[0].periodicity_hz) else np.nan
        fiducial_peaks = _detect_fiducial_peaks(y_detect, fs, stage_cfg, expected_freq)
        if fiducial_peaks.size >= stage_cfg.min_events:
            pattern_peak = int(np.argmax(pattern))
            starts_from_fiducials = np.clip(fiducial_peaks - pattern_peak, 0, max(n - m, 0))
            fits = [
                _fit_pattern_at_start(y_detect, pattern, int(peak), fs, stage_cfg)
                for peak in _event_subset(starts_from_fiducials, stage_cfg.max_fit_events)
            ]
            fiducial_event_indices = _event_subset(fiducial_peaks, stage_cfg.max_fit_events)
            fiducial_event_times = fiducial_event_indices / fs

    component = _build_component_from_fits(n, pattern, fits, fs, stage_cfg)
    residual = y - component
    event_times, event_indices, event_amplitudes = _events_from_fits(fits, fs)
    if fiducial_event_times is not None and fiducial_event_indices is not None:
        event_times = fiducial_event_times.astype(float)
        event_indices = fiducial_event_indices.astype(int)
        event_amplitudes = _amplitudes_for_event_count(event_amplitudes, event_times.size)
    inst_freq = _instantaneous_frequency(event_times, n, fs, global_cfg.smooth_frequency_s)
    inst_amp = _instantaneous_amplitude(
        event_times,
        event_amplitudes,
        n,
        fs,
        global_cfg.smooth_amplitude_s,
    )

    return PatternStageResult(
        name=stage_cfg.name,
        input_signal=y.copy(),
        component=component,
        residual_signal=residual,
        pattern=pattern,
        pattern_time_s=np.arange(pattern.size, dtype=float) / fs,
        event_times_s=event_times,
        event_start_indices=event_indices,
        instantaneous_frequency_hz=inst_freq,
        instantaneous_amplitude=inst_amp,
        correlation=corr,
        correlation_time_s=corr_t,
        fits=fits,
        candidates=candidates,
    )


def _stage_detection_signal(y: np.ndarray, fs: float, cfg: PatternStageConfig) -> np.ndarray:
    if cfg.highpass_cutoff_hz is None:
        return y
    cutoff = cfg.highpass_cutoff_hz
    if cutoff <= 0.0 or cutoff >= 0.45 * fs or y.size < 16:
        return y
    b, a = scipy.signal.butter(2, cutoff, btype="highpass", fs=fs)
    return scipy.signal.filtfilt(b, a, y)


def _detect_fiducial_peaks(
    y: np.ndarray,
    fs: float,
    cfg: PatternStageConfig,
    expected_freq_hz: float,
) -> np.ndarray:
    f_low, f_high = cfg.frequency_band_hz
    if not np.isfinite(expected_freq_hz):
        expected_freq_hz = 0.5 * (f_low + f_high)
    expected_freq_hz = float(np.clip(expected_freq_hz, f_low, f_high))
    distance = max(1, int(round(0.68 * fs / expected_freq_hz)))
    positive = y[np.isfinite(y)]
    if positive.size == 0:
        return np.array([], dtype=int)
    height = float(np.nanpercentile(positive, 58.0))
    prominence = max(0.05 * float(np.nanstd(positive)), 1e-5)
    peaks, _ = scipy.signal.find_peaks(
        y,
        height=height,
        prominence=prominence,
        distance=distance,
    )
    return peaks.astype(int)


def normalized_sliding_correlation(
    y: np.ndarray,
    template: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = _as_float_1d(y)
    p = _normalize_pattern(template)
    m = p.size
    if y.size < m:
        return np.array([], dtype=float), np.array([], dtype=float)

    p = p - np.mean(p)
    p_energy = float(np.sum(p**2))
    raw = scipy.signal.correlate(y, p, mode="valid", method="auto")
    kernel = np.ones(m, dtype=float)
    win_sum = np.convolve(y, kernel, mode="valid")
    win_sum2 = np.convolve(y**2, kernel, mode="valid")
    win_energy = win_sum2 - (win_sum**2 / m)
    denom = np.sqrt(np.maximum(win_energy * p_energy, 1e-12))
    corr = raw / denom
    corr[~np.isfinite(corr)] = 0.0
    corr_t = (np.arange(corr.size, dtype=float) + (m - 1) / 2.0) / fs
    return corr, corr_t


def _fit_pattern_at_start(
    y: np.ndarray,
    pattern: np.ndarray,
    start_index: int,
    fs: float,
    cfg: PatternStageConfig,
) -> PatternFit:
    pattern = _normalize_pattern(pattern)
    n = y.size
    m = pattern.size
    pattern_duration_s = m / fs
    margin = int(round(cfg.fit_margin_s * fs))
    local_start = max(0, start_index - margin)
    local_end = min(n, start_index + m + margin)
    local_indices = np.arange(local_start, local_end)
    rel_t = (local_indices - start_index) / fs
    y_local = y[local_start:local_end]
    p_curve = scipy.interpolate.interp1d(
        np.linspace(0.0, 1.0, m),
        pattern,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )

    def residual(params: np.ndarray) -> np.ndarray:
        tau_s, stretch = params
        duration = pattern_duration_s * stretch
        u = (rel_t - tau_s) / duration
        mask = (0.0 <= u) & (u <= 1.0)
        out = np.zeros_like(y_local)
        if np.count_nonzero(mask) < max(5, int(0.35 * m)):
            out[:] = 1e3
            return out
        p = p_curve(u[mask])
        x = np.column_stack((p, u[mask] * p, np.ones(np.count_nonzero(mask))))
        beta, *_ = np.linalg.lstsq(x, y_local[mask], rcond=None)
        model = x @ beta
        mask_fraction = max(np.count_nonzero(mask) / max(m, 1), 0.2)
        out[mask] = (model - y_local[mask]) / np.sqrt(mask_fraction)
        return out

    tau_bound = cfg.fit_margin_s
    bounds = ([-tau_bound, cfg.stretch_bounds[0]], [tau_bound, cfg.stretch_bounds[1]])
    try:
        res = scipy.optimize.least_squares(
            residual,
            x0=np.array([0.0, 1.0]),
            bounds=bounds,
            max_nfev=80,
            x_scale=np.array([max(tau_bound, 1e-3), 1.0]),
        )
        tau_s, stretch = map(float, res.x)
        beta0, beta1, offset, amplitude, cost = _linear_fit_params_for_solution(
            y_local,
            rel_t,
            p_curve,
            tau_s,
            pattern_duration_s * stretch,
        )
        success = bool(res.success and np.isfinite(cost))
    except (ValueError, np.linalg.LinAlgError):
        tau_s = 0.0
        stretch = 1.0
        beta0 = 0.0
        beta1 = 0.0
        offset = 0.0
        amplitude = np.nan
        cost = np.inf
        success = False

    return PatternFit(
        start_index_guess=int(start_index),
        start_time_s=float(start_index / fs + tau_s),
        duration_s=float(pattern_duration_s * stretch),
        tau_s=float(tau_s),
        stretch=float(stretch),
        beta0=float(beta0),
        beta1=float(beta1),
        offset=float(offset),
        amplitude=float(amplitude),
        cost=float(cost),
        success=success,
    )


def _linear_fit_params_for_solution(
    y_local: np.ndarray,
    rel_t: np.ndarray,
    p_curve: scipy.interpolate.interp1d,
    tau_s: float,
    duration_s: float,
) -> tuple[float, float, float, float, float]:
    u = (rel_t - tau_s) / duration_s
    mask = (0.0 <= u) & (u <= 1.0)
    if np.count_nonzero(mask) < 5:
        return 0.0, 0.0, 0.0, np.nan, np.inf
    p = p_curve(u[mask])
    x = np.column_stack((p, u[mask] * p, np.ones(np.count_nonzero(mask))))
    beta, *_ = np.linalg.lstsq(x, y_local[mask], rcond=None)
    model = x @ beta
    component = (beta[0] + beta[1] * u[mask]) * p
    amplitude = robust_half_peak_to_peak(component)
    cost = float(np.mean((model - y_local[mask]) ** 2))
    return float(beta[0]), float(beta[1]), float(beta[2]), amplitude, cost


def _average_fitted_patterns(
    y: np.ndarray,
    reference_pattern: np.ndarray,
    fits: list[PatternFit],
    fs: float,
    output_len: int,
) -> np.ndarray | None:
    reference_pattern = _normalize_pattern(reference_pattern)
    u_common = np.linspace(0.0, 1.0, output_len)
    stored = []
    n = y.size

    for fit in fits:
        if not fit.success or not np.isfinite(fit.amplitude) or fit.amplitude <= 1e-6:
            continue
        start = max(0, int(np.floor(fit.start_time_s * fs)))
        end = min(n, int(np.ceil((fit.start_time_s + fit.duration_s) * fs)) + 1)
        if end - start < 5:
            continue
        idx = np.arange(start, end)
        u = (idx / fs - fit.start_time_s) / fit.duration_s
        mask = (0.0 <= u) & (u <= 1.0)
        idx = idx[mask]
        u = u[mask]
        amp_env = fit.beta0 + fit.beta1 * u
        valid = np.abs(amp_env) > 0.15 * max(abs(fit.beta0), abs(fit.beta0 + fit.beta1), 1e-6)
        if np.count_nonzero(valid) < 5:
            continue
        u = u[valid]
        idx = idx[valid]
        amp_env = amp_env[valid]
        normalized = (y[idx] - fit.offset) / amp_env
        if np.count_nonzero(np.isfinite(normalized)) < 5:
            continue
        order = np.argsort(u)
        sample = np.interp(u_common, u[order], normalized[order])
        sample = _normalize_pattern(sample)
        if np.dot(sample, reference_pattern) < 0.0:
            sample = -sample
        stored.append(sample)

    if not stored:
        return None
    avg = np.nanmedian(np.vstack(stored), axis=0)
    return _normalize_pattern(avg)


def _build_component_from_fits(
    n: int,
    pattern: np.ndarray,
    fits: list[PatternFit],
    fs: float,
    cfg: PatternStageConfig,
) -> np.ndarray:
    pattern = _normalize_pattern(pattern)
    p_curve = scipy.interpolate.interp1d(
        np.linspace(0.0, 1.0, pattern.size),
        pattern,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    acc = np.zeros(n, dtype=float)
    weight = np.zeros(n, dtype=float)

    for fit in fits:
        if not fit.success:
            continue
        start = max(0, int(np.floor(fit.start_time_s * fs)))
        end = min(n, int(np.ceil((fit.start_time_s + fit.duration_s) * fs)) + 1)
        if end <= start:
            continue
        idx = np.arange(start, end)
        u = (idx / fs - fit.start_time_s) / fit.duration_s
        mask = (0.0 <= u) & (u <= 1.0)
        idx = idx[mask]
        u = u[mask]
        if idx.size == 0:
            continue
        component = (fit.beta0 + fit.beta1 * u) * p_curve(u)
        taper = _edge_taper(u, cfg.subtraction_taper_fraction)
        acc[idx] += component * taper
        weight[idx] += taper

    component = np.zeros(n, dtype=float)
    valid = weight > 1e-9
    component[valid] = acc[valid] / np.maximum(weight[valid], 1.0)
    return component


def _edge_taper(u: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return np.ones_like(u)
    w = np.ones_like(u, dtype=float)
    left = u < fraction
    right = u > 1.0 - fraction
    w[left] = 0.5 - 0.5 * np.cos(np.pi * u[left] / fraction)
    w[right] = 0.5 - 0.5 * np.cos(np.pi * (1.0 - u[right]) / fraction)
    return np.clip(w, 0.0, 1.0)


def _events_from_fits(fits: list[PatternFit], fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    good = [fit for fit in fits if fit.success and np.isfinite(fit.start_time_s)]
    good.sort(key=lambda fit: fit.start_time_s)
    times = np.array([fit.start_time_s for fit in good], dtype=float)
    indices = np.rint(times * fs).astype(int)
    amplitudes = np.array([fit.amplitude for fit in good], dtype=float)
    return times, indices, amplitudes


def _amplitudes_for_event_count(amplitudes: np.ndarray, event_count: int) -> np.ndarray:
    if event_count <= 0:
        return np.array([], dtype=float)
    if amplitudes.size == event_count:
        return amplitudes
    if amplitudes.size == 0:
        return np.full(event_count, np.nan, dtype=float)
    old_x = np.linspace(0.0, 1.0, amplitudes.size)
    new_x = np.linspace(0.0, 1.0, event_count)
    return np.interp(new_x, old_x, amplitudes)


def _instantaneous_frequency(
    event_times_s: np.ndarray,
    n: int,
    fs: float,
    smooth_s: float,
) -> np.ndarray:
    out = np.full(n, np.nan, dtype=float)
    if event_times_s.size < 2:
        return out
    intervals = np.diff(event_times_s)
    valid = intervals > 1e-6
    if np.count_nonzero(valid) == 0:
        return out
    freq = 1.0 / intervals[valid]
    freq_t = 0.5 * (event_times_s[:-1][valid] + event_times_s[1:][valid])
    t = np.arange(n, dtype=float) / fs
    right = float(freq[-1])
    interp = np.interp(t, freq_t, freq, left=np.nan, right=right)
    interp[t < freq_t[0]] = np.nan
    return _smooth_nan_series(interp, fs, smooth_s)


def _instantaneous_amplitude(
    event_times_s: np.ndarray,
    amplitudes: np.ndarray,
    n: int,
    fs: float,
    smooth_s: float,
) -> np.ndarray:
    out = np.full(n, np.nan, dtype=float)
    valid_events = np.isfinite(event_times_s) & np.isfinite(amplitudes)
    if np.count_nonzero(valid_events) == 0:
        return out
    times = event_times_s[valid_events]
    amps = amplitudes[valid_events]
    order = np.argsort(times)
    times = times[order]
    amps = amps[order]
    t = np.arange(n, dtype=float) / fs
    interp = np.interp(t, times, amps, left=np.nan, right=float(amps[-1]))
    interp[t < times[0]] = np.nan
    return _smooth_nan_series(interp, fs, smooth_s)


def _smooth_nan_series(x: np.ndarray, fs: float, smooth_s: float) -> np.ndarray:
    if smooth_s <= 0.0:
        return x
    valid = np.isfinite(x)
    if not np.any(valid):
        return x
    filled = x.copy()
    first = int(np.argmax(valid))
    filled[:first] = x[first]
    if not np.all(valid):
        idx = np.arange(x.size)
        filled[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    win = max(3, int(round(smooth_s * fs)))
    if win % 2 == 0:
        win += 1
    if win >= x.size:
        return x
    smoothed = scipy.signal.savgol_filter(filled, win, polyorder=1, mode="interp")
    smoothed[:first] = np.nan
    return smoothed


def _candidate_starts(n: int, m: int, count: int, fs: float) -> np.ndarray:
    if n <= m:
        return np.array([0], dtype=int)
    fractions = np.array([0.00, 0.025, 0.06, 0.12, 0.20, 0.31, 0.43, 0.56, 0.69, 0.81, 0.91, 0.98])
    if count > fractions.size:
        extra = np.linspace(0.04, 0.96, count - fractions.size + 2)[1:-1]
        fractions = np.unique(np.concatenate((fractions, extra)))
    starts = np.rint(fractions[:count] * (n - m)).astype(int)
    starts = starts[(0 <= starts) & (starts <= n - m)]
    return np.unique(starts)


def _find_correlation_peaks(
    corr: np.ndarray,
    fs: float,
    cfg: PatternStageConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if corr.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    f_low, f_high = cfg.frequency_band_hz
    distance = max(1, int(round(cfg.peak_distance_factor * fs / f_high)))
    finite = corr[np.isfinite(corr)]
    if finite.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    height = max(cfg.min_corr_peak, float(np.nanpercentile(finite, 70.0)))
    height = min(height, 0.92 * float(np.nanmax(finite)))
    peaks, props = scipy.signal.find_peaks(
        corr,
        height=height,
        distance=distance,
        prominence=cfg.peak_prominence,
    )
    if peaks.size < cfg.min_events:
        peaks, props = scipy.signal.find_peaks(
            corr,
            height=cfg.min_corr_peak * 0.65,
            distance=distance,
            prominence=cfg.peak_prominence * 0.5,
        )
    heights = props.get("peak_heights", corr[peaks])
    return peaks.astype(int), np.asarray(heights, dtype=float)


def _periodicity_from_peaks(peaks: np.ndarray, fs: float) -> tuple[float, float]:
    if peaks.size < 2:
        return np.nan, np.inf
    intervals = np.diff(peaks) / fs
    intervals = intervals[intervals > 0.0]
    if intervals.size == 0:
        return np.nan, np.inf
    median_interval = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - median_interval)))
    cv = 1.4826 * mad / max(median_interval, 1e-9)
    return 1.0 / median_interval, cv


def _candidate_score(
    freq_hz: float,
    cv: float,
    heights: np.ndarray,
    event_count: int,
    cfg: PatternStageConfig,
) -> float:
    if not np.isfinite(freq_hz) or not np.isfinite(cv):
        return -np.inf
    f_low, f_high = cfg.frequency_band_hz
    in_band = f_low <= freq_hz <= f_high
    band_factor = 1.0 if in_band else 0.2
    height_score = float(np.nanmedian(heights)) if heights.size else 0.0
    count_score = min(event_count / max(cfg.min_events, 1), 2.5)
    return band_factor * height_score * count_score / (1.0 + 3.0 * cv)


def _select_candidates(
    candidates: list[CandidatePattern],
    cfg: PatternStageConfig,
) -> list[CandidatePattern]:
    f_low, f_high = cfg.frequency_band_hz
    valid = [
        c
        for c in candidates
        if c.score > -np.inf
        and c.peaks.size >= cfg.min_events
        and f_low <= c.periodicity_hz <= f_high
    ]
    if not valid:
        return []
    valid.sort(key=lambda c: c.score, reverse=True)
    best_score = valid[0].score
    best_freq = valid[0].periodicity_hz
    selected = [
        c
        for c in valid
        if c.score >= 0.45 * best_score
        and abs(c.periodicity_hz - best_freq) / max(best_freq, 1e-9) < 0.35
    ]
    return selected[: cfg.max_templates_to_average]


def _event_subset(peaks: np.ndarray, max_events: int | None) -> np.ndarray:
    if max_events is None or peaks.size <= max_events:
        return peaks
    idx = np.linspace(0, peaks.size - 1, max_events)
    return peaks[np.unique(np.rint(idx).astype(int))]


def _normalize_pattern(x: np.ndarray) -> np.ndarray:
    p = np.asarray(x, dtype=float).copy()
    if p.size == 0:
        return p
    p = p - np.nanmean(p)
    scale = robust_half_peak_to_peak(p)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(p))
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(p)
    return p / scale


def _as_float_1d(sig: np.ndarray) -> np.ndarray:
    y = np.asarray(sig, dtype=float)
    if y.ndim != 1:
        raise ValueError("sig must be a one-dimensional array")
    if y.size < 8:
        raise ValueError("sig is too short for pattern extraction")
    return y


def _plot_stage_diagnostics(
    stage: PatternStageResult,
    t: np.ndarray,
    fs: float,
    output_path: Path | None,
) -> list[object]:
    import matplotlib.pyplot as plt

    figures: list[object] = []
    ranked = sorted(stage.candidates, key=lambda c: c.score, reverse=True)
    best = ranked[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t, stage.input_signal, lw=0.9)
    for candidate in ranked[: min(5, len(ranked))]:
        start = candidate.template_start_index
        end = start + candidate.template.size
        axes[0].axvspan(start / fs, end / fs, alpha=0.16)
        axes[0].text(
            (start + end) / (2.0 * fs),
            np.nanmax(stage.input_signal),
            f"{candidate.periodicity_hz:.2f} Hz",
            ha="center",
            va="top",
            fontsize=8,
        )
    axes[0].set_title(f"{stage.name}: deterministic candidate windows")
    axes[0].grid(True, alpha=0.3)

    for candidate in ranked[: min(5, len(ranked))]:
        axes[1].plot(candidate.correlation_time_s, candidate.correlation, lw=0.8)
    axes[1].plot(best.correlation_time_s[best.peaks], best.correlation[best.peaks], "x", label="best peaks")
    axes[1].set_title("Normalized sliding correlations and detected matches")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(stage.pattern_time_s, stage.pattern, lw=2.0, label="final average pattern")
    if best.average_pattern is not None:
        axes[2].plot(stage.pattern_time_s, best.average_pattern, "--", label="best-window average")
    axes[2].plot(np.arange(best.template.size) / fs, best.template, alpha=0.45, label="best window pattern")
    axes[2].set_title("Window pattern, stored-pattern average, and final consensus")
    axes[2].set_xlabel("pattern time [s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path, f"10_{stage.name}_candidate_correlation.png")
    figures.append(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    overlay_count = min(18, len(stage.fits))
    u_common = np.linspace(0.0, 1.0, stage.pattern.size)
    for fit in stage.fits[:overlay_count]:
        start = max(0, int(np.floor(fit.start_time_s * fs)))
        end = min(stage.input_signal.size, int(np.ceil((fit.start_time_s + fit.duration_s) * fs)) + 1)
        if end - start < 5:
            continue
        idx = np.arange(start, end)
        u = (idx / fs - fit.start_time_s) / fit.duration_s
        amp_env = fit.beta0 + fit.beta1 * u
        valid = (0.0 <= u) & (u <= 1.0) & (np.abs(amp_env) > 1e-6)
        if np.count_nonzero(valid) >= 5:
            y_norm = (stage.input_signal[idx[valid]] - fit.offset) / amp_env[valid]
            order = np.argsort(u[valid])
            axes[0].plot(u_common, np.interp(u_common, u[valid][order], y_norm[order]), alpha=0.25)
    axes[0].plot(u_common, stage.pattern, color="black", lw=2.0, label="average")
    axes[0].set_title("Stored normalized fitted patterns")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, stage.input_signal, lw=0.7, alpha=0.55, label="stage input")
    axes[1].plot(t, stage.component, lw=1.2, label="fitted component")
    for fit in stage.fits[: min(10, len(stage.fits))]:
        axes[1].axvline(fit.start_time_s, color="tab:red", alpha=0.18)
    axes[1].set_title("Fitted average pattern placed back on the signal")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, stage.input_signal, lw=0.7, label="before")
    axes[2].plot(t, stage.residual_signal, lw=0.9, label="after suppression")
    axes[2].set_title("Suppression result")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    if stage.fits:
        sample = stage.fits[min(2, len(stage.fits) - 1)]
        text = (
            f"example fit: stretch={sample.stretch:.3f}, "
            f"tau={sample.tau_s:.3f}s, beta0={sample.beta0:.3g}, "
            f"beta1={sample.beta1:.3g}, offset={sample.offset:.3g}"
        )
        fig.text(0.01, 0.01, text, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save_fig(fig, output_path, f"20_{stage.name}_fit_average_suppress.png")
    figures.append(fig)

    return figures


def _save_fig(fig: object, output_path: Path | None, filename: str) -> None:
    if output_path is None:
        return
    fig.savefig(output_path / filename, dpi=150)


def _print_stage_summary(result: VitalPatternExtractionResult) -> None:
    for stage in result.stages:
        valid_freq = stage.instantaneous_frequency_hz[np.isfinite(stage.instantaneous_frequency_hz)]
        valid_amp = stage.instantaneous_amplitude[np.isfinite(stage.instantaneous_amplitude)]
        freq_text = "nan" if valid_freq.size == 0 else f"{np.nanmedian(valid_freq):.4g}"
        amp_text = "nan" if valid_amp.size == 0 else f"{np.nanmedian(valid_amp):.4g}"
        print(
            f"{stage.name}: events={stage.event_times_s.size}, "
            f"median_frequency_hz={freq_text}, median_amplitude={amp_text}, "
            f"pattern_samples={stage.pattern.size}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sliding-window correlation plus least-squares pattern fitting for "
            "nonstationary respiration and heart displacement signals."
        )
    )
    parser.add_argument("--fs", type=float, default=20.0, help="Sampling frequency for the synthetic benchmark.")
    parser.add_argument("--duration", type=float, default=60.0, help="Synthetic benchmark duration in seconds.")
    parser.add_argument("--heart-scale", type=float, default=0.1, help="Heart displacement scale in the synthetic signal.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="White-noise standard deviation added to the synthetic signal.")
    parser.add_argument("--br-mod-scale", type=float, default=0.15, help="Respiration frequency modulation depth.")
    parser.add_argument("--hr-mod-scale", type=float, default=0.20, help="Heart frequency modulation depth.")
    parser.add_argument("--seed", type=int, default=1234, help="Synthetic noise RNG seed.")
    parser.add_argument("--grace-s", type=float, default=10.0, help="Metric grace period in seconds.")
    parser.add_argument("--demo", action="store_true", help="Create explanatory matplotlib figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sliding_pattern_demo_output"),
        help="Where demo figures are written.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively after creating them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = benchmark_challenge_signal(
        fs=args.fs,
        duration_s=args.duration,
        heart_scale=args.heart_scale,
        noise_std=args.noise_std,
        br_mod_scale=args.br_mod_scale,
        hr_mod_scale=args.hr_mod_scale,
        seed=args.seed,
    )
    signal = np.asarray(data["signal"])
    fs = float(data["fs"])
    if args.demo:
        result, _ = introductory_pattern_extraction_demo(
            signal,
            fs,
            true_frequency_by_stage={
                "breath": np.asarray(data["breath_frequency_hz"]),
                "heart": np.asarray(data["heart_frequency_hz"]),
            },
            true_amplitude_by_stage={
                "breath": float(data["breath_amplitude"]),
                "heart": float(data["heart_amplitude"]),
            },
            output_dir=args.output_dir,
            show=args.show,
        )
        print(f"wrote demo figures to: {args.output_dir}")
    else:
        result = extract_vital_patterns(signal, fs)

    _print_stage_summary(result)
    metrics = score_extraction_against_truth(result, data, grace_s=args.grace_s)
    for key, value in metrics.items():
        print(f"{key}: {value:.5g}")


if __name__ == "__main__":
    main()
