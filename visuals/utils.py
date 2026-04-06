from PySide6.QtCore import QObject, QEvent
from PySide6.QtWidgets import QSlider

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QColor


from superqt import QRangeSlider

class SliderWheelFilter(QObject):
    def eventFilter(self, obj, event):
        if isinstance(obj, QSlider) and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()

            if isinstance(obj,QRangeSlider): # for the range slider
                pos = event.position().x() # returns in pixels i guess
                width = obj.width()

                min_val = obj.minimum()
                max_val = obj.maximum()

                coresponding_value = min_val + (max_val - min_val) * (pos / width)

                low,high = obj.value()

                step = obj.singleStep()

                if coresponding_value <= low: # targeting the low part
                    if delta > 0:
                        obj.setValue((low+step,high))
                    elif delta < 0:
                        obj.setValue((low-step,high))
                elif coresponding_value >= high: # targeting the high part
                    if delta > 0:
                        obj.setValue((low,high+step))
                    elif delta < 0:
                        obj.setValue((low,high-step))
                else: # targeting the range
                    if delta > 0:
                        obj.setValue((low+step,high+step))
                    elif delta < 0:
                        obj.setValue((low-step,high-step))
                
                if isinstance(obj,HighlightRangeSlider):
                    if coresponding_value <= low: # targeting the low part
                        obj.highlight("low")
                    elif coresponding_value >= high: # targeting the high part
                        obj.highlight("high")
                    else: # targeting the range
                       obj.highlight("middle")


            else: # normal slider    
                if delta > 0:
                    obj.setValue(obj.value() + obj.singleStep())
                elif delta < 0:
                    obj.setValue(obj.value() - obj.singleStep())

            return True  # block default behavior

        return super().eventFilter(obj, event)




class HighlightRangeSlider(QRangeSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._highlight = None  # "low", "middle", "high"
        self._highlight_color_og = QColor(255, 200, 0, 120)  # translucent yellow
        # simple decay animation:
        self._highlight_color = QColor(self._highlight_color_og)
        self._highlightDecayTicks = 0
        self._decayDuration = 80
        self.decayTimer = QTimer()
        self.decayTimer.timeout.connect(self.decay_highlight)

    # --- public API ---
    def highlight(self, part, duration=40):
        self._highlightDecayTicks = 0
        self._highlight_color = QColor(self._highlight_color_og)
        self._highlight = part
        self._decayDuration = duration
        self.update()
        self.decayTimer.stop()

        self.decayTimer.start(200) #first duration is longer to smooth stuff out
    
    def decay_highlight(self):
        
        self._highlight_color.setAlpha(min(255,int(self._highlight_color.alpha() / ((1.04)**self._highlightDecayTicks) )) )
        self._highlightDecayTicks +=1
        # print(f"Alfa {self._highlight_color.alpha()}")

        if(self._highlightDecayTicks <= 9):
            self.decayTimer.start(self._decayDuration)
        else:
            self.decayTimer.stop()
            self.clear_highlight()
        self.update()

    def clear_highlight(self):
        self._highlight = None
        self.update()
        self._highlightDecayTicks = 0
        self._highlight_color = QColor(self._highlight_color_og)

    # --- painting ---
    def paintEvent(self, event):
        # let the base class draw everything first
        super().paintEvent(event)

        if not self._highlight:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # geometry
        rect = self.rect()
        min_val = self.minimum()
        max_val = self.maximum()
        low, high = self.value()

        # helper: value → x position
        def val_to_x(val):
            return int((val - min_val) / (max_val - min_val) * rect.width())

        x_low = val_to_x(low)
        x_high = val_to_x(high)

        # decide region
        if self._highlight == "low":
            highlight_rect = rect.adjusted(0, 0, x_low - rect.width(), 0)

        elif self._highlight == "middle":
            highlight_rect = rect.adjusted(x_low, 0, -(rect.width() - x_high), 0)

        elif self._highlight == "high":
            highlight_rect = rect.adjusted(x_high, 0, 0, 0)

        else:
            return

        painter.fillRect(highlight_rect, self._highlight_color)