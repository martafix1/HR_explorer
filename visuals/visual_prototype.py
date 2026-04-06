import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QLabel, QSplitter
)
from PySide6.QtCore import Qt

import pyqtgraph as pg


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Splitter + Plot Example")
        self.resize(900, 600)

        # --- Splitter ---
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle {
            background-color: #e0e0e0;
        }
        QSplitter::handle:hover {
            background-color: #c0c0c0;
        }
        """)

        # --- Top: Control section ---
        control_widget = QWidget()
        control_widget.setStyleSheet("background-color: #2b2b2b; color: white;")
        control_layout = QVBoxLayout(control_widget)
        control_layout.addWidget(QLabel("Control panel (put buttons here)"))

        # --- Bottom: Plot section ---
        display_widget = QWidget()
        display_widget.setStyleSheet("background-color: #1e1e1e;")
        display_layout = QVBoxLayout(display_widget)

        # Create a pyqtgraph plot
        plot_widget = pg.PlotWidget()
        display_layout.addWidget(plot_widget)

        # Generate sine wave
        x = np.linspace(0, 10, 500)
        y = np.sin(x)

        plot_widget.plot(x, y, pen="c")

        # Add widgets to splitter
        splitter.addWidget(control_widget)
        splitter.addWidget(display_widget)

        # Set stretch (important for proportions)
        splitter.setStretchFactor(0, 2)  # top
        splitter.setStretchFactor(1, 3)  # bottom

        # Also set initial sizes explicitly (more reliable)
        splitter.setSizes([200, 400])  # 2:3 ratio

        self.setCentralWidget(splitter)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())