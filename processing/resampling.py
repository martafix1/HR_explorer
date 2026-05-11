

import numpy as np
import scipy




def interpolate_missing_frames(signal,frames):

    frames_full = np.arange(frames[0],frames[-1],1)

    data_interp = np.interp(
        frames_full,
        frames,
        signal
    )

    return frames_full, data_interp


def resample_HR_to_frames(HRs : dict, fs = 20):

    HR_dict = HRs
    pairs = {}

    for key in HR_dict:
        if key.startswith("hr_") and not key.endswith("_delays_ms"):
            delay_key = f"{key}_delays_ms"

            if delay_key in HR_dict:
                pairs[key] = {
                    "signal": HR_dict[key],
                    "delays_ms": HR_dict[delay_key],
                }

    print(pairs)

    pairs_resampled = {}

    for key,val in pairs.items():

        samples = val["signal"]
        delays_ms = val["delays_ms"]
        times_s = np.concatenate([[0.0],np.cumsum(delays_ms) / 1000.0])

        times_frameTimeBase = times_s * fs

        frames = np.rint(np.arange(0,times_frameTimeBase[-1])).astype(int)  # support for signal in frames

        # zero-order hold interpolator
        zoh = scipy.interpolate.interp1d(
            times_frameTimeBase,
            samples,
            kind='previous',
            bounds_error=False,
            fill_value=(samples[0], samples[-1]),
        )

        resampled = zoh(frames) /60 # to turn it into Hz
        pairs_resampled[key] = {"frames": frames, "hr": resampled}

    return pairs_resampled