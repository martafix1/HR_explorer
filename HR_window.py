import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QComboBox, QSizePolicy
)
from PySide6.QtCore import Qt

import numpy as np
import pyqtgraph as pg


class STFT_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Window")

        main_layout = QVBoxLayout(self)

        # ===== TOP CONTROL BLOCK =====
        control_block = QWidget()
        control_block.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        control_layout = QHBoxLayout(control_block)

        # ---- LEFT: 3x2 sliders ----
        slider_grid = QGridLayout()

        self.sliders = {}

        names = [
            "Frames start", "Frames end",
            "Param C", "Param D",
            "Param E", "Param F"
        ]

        for i, name in enumerate(names):
            row = i // 2
            col = i % 2

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)

            label = QLabel(f"{name}: 0")

            # update label on change
            slider.valueChanged.connect(
                lambda value, l=label, n=name: l.setText(f"{n}: {value}")
            )

            # container for slider + label
            vbox = QVBoxLayout()
            vbox.addWidget(slider)
            vbox.addWidget(label)

            slider_grid.addLayout(vbox, row, col)

            self.sliders[name] = slider

        control_layout.addLayout(slider_grid)

        # ---- RIGHT: 3 dropdowns ----
        dropdown_layout = QVBoxLayout()

        self.dropdowns = []

        for i in range(3):
            combo = QComboBox()
            combo.addItems(["Option 1", "Option 2", "Option 3"])
            dropdown_layout.addWidget(combo)
            self.dropdowns.append(combo)

        dropdown_layout.addStretch()  # keeps them compact at top
        control_layout.addLayout(dropdown_layout)

        # ===== HEATMAP AREA (placeholder) =====
        self.setup_heatmap()

        # ===== ADD TO MAIN =====
        main_layout.addWidget(control_block)
        main_layout.addWidget(self.heatmap)

        # Make control block minimal height
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)


    def initFromData(self):

        pass
    
    def do_STFT(self,frames,sliders):
        
        


        pass

    def setup_heatmap(self):
        # Create ImageView (has zoom, pan, histogram built-in)
        self.heatmap = pg.ImageView()

        # Optional: nicer color map
        colormap = pg.colormap.get("viridis")  # or "inferno", "plasma"
        self.heatmap.setColorMap(colormap)


        # Initial data
        data = np.random.rand(20,40)
        self.heatmap.setImage(data)

        return self.heatmap
    
    def update_heatmap(self,data):
        data = np.random.rand(*self.heatmap_shape)
        self.heatmap.setImage(data, autoLevels=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = STFT_Window()
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())