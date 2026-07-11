
"""
stream_eeg.py
 
Streams live EEG data from a Muse headband via Lab Streaming Layer (LSL).
 
SETUP (one-time):
    pip install muselsl pylsl
 
HOW TO RUN (two terminals):
    Terminal 1 — start the Muse LSL stream (connects over Bluetooth):
        muselsl stream
 
    Terminal 2 — run this script to read the stream:
        python stream_eeg.py
 
If `muselsl stream` can't find your headset, first run `muselsl list`
to confirm it's discoverable (make sure the Muse is on and not already
paired to another app, e.g. the Muse phone app).
"""
 
from pylsl import StreamInlet, resolve_byprop
from collections import deque
import time
import signal
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
 

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10", "Right AUX"]
WINDOW_SECONDS = 2


def connect_to_muse(ti=10):
    streams = resolve_byprop("type", "EEG", timeout=ti)
    if not streams:
        raise RuntimeError("No streams found")
    else:
        inlet = StreamInlet(streams[0])
        return inlet


def main(inlet):
    app = QtWidgets.QApplication(sys.argv)
    dashboard = EEGDashboard(inlet)
    dashboard.show()
    sys.exit(app.exec())



class EEGDashboard(QtWidgets.QMainWindow):
    def __init__(self, inlet):
        super().__init__()
        self.inlet = inlet
        self.num_samples = int(inlet.info().nominal_srate() * WINDOW_SECONDS)
  
        self.data = {name: deque(maxlen=self.num_samples) for name in CHANNEL_NAMES}

        self.timestamps = deque(maxlen=self.num_samples)

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(33)

    def _build_ui(self):
        self.setWindowTitle("EEG Dashboard")
        self.resize(1000, 700)

        glw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(glw)

        self.curves = {}  

        for i, name in enumerate(CHANNEL_NAMES):
            p = glw.addPlot(row=i, col=0)
            p.setLabel("left", name)
            curve = p.plot()
            self.curves[name] = curve

    def _update(self):
        samples, timestamp = self.inlet.pull_chunk(timeout=0.0)

        if not samples:
            return  
        
        num_samples = len(samples)

        for i in range(0,num_samples):
            sample = samples[i]
            t = timestamp[i]
            for name, value in zip(CHANNEL_NAMES, sample):
                self.data[name].append(value)
            self.timestamps.append(t)

        # redraw each curve with the latest buffer contents
        for name in CHANNEL_NAMES:
            self.curves[name].setData(list(self.timestamps), list(self.data[name]))

if __name__ == "__main__":
    inlet = connect_to_muse(ti=10)
    main(inlet)
