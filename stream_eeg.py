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
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

import scipy.signal as sps
import numpy as np


CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
WINDOW_SECONDS = 2


def connect_to_muse(ti=10):
    streams = resolve_byprop("type", "EEG", timeout=ti)
    if not streams:
        raise RuntimeError("No streams found")
    else:
        inlet = StreamInlet(streams[0])
        return inlet


def main():
    inlet = connect_to_muse(ti=10)
    app = QtWidgets.QApplication(sys.argv)
    dashboard = EEGDashboard(inlet)
    dashboard.show()
    sys.exit(app.exec())


class EEGDashboard(QtWidgets.QMainWindow):
    def __init__(self, inlet):
        super().__init__()
        self.inlet = inlet
        self.fs = self.inlet.info().nominal_srate()
        self.num_samples = int(self.fs * WINDOW_SECONDS)

        self.data = {name: deque(maxlen=self.num_samples) for name in CHANNEL_NAMES}
        self.filtered_data = {name: deque(maxlen=self.num_samples) for name in CHANNEL_NAMES}
        self.timestamps = deque(maxlen=self.num_samples)

        # Combined notch (60Hz) + lowpass (40Hz) filter, applied as one
        # cascaded causal IIR filter so we only need one persistent
        # filter-state (zi) per channel instead of two.
        notch_b, notch_a = sps.iirnotch(60, 30, self.fs)
        low_b, low_a = sps.butter(4, 40, btype="low", fs=self.fs)
        self.filt_b, self.filt_a = self._cascade(notch_b, notch_a, low_b, low_a)

        zero_state = np.zeros(max(len(self.filt_a), len(self.filt_b)) - 1)
        self.filt_zi = {name: zero_state.copy() for name in CHANNEL_NAMES}

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(33)

    @staticmethod
    def _cascade(b1, a1, b2, a2):
        """Combine two IIR filters into one set of (b, a) coefficients
        so they can be applied in a single lfilter call with one zi."""
        b = np.convolve(b1, b2)
        a = np.convolve(a1, a2)
        return b, a

    def _build_ui(self):
        self.setWindowTitle("EEG Dashboard")
        self.resize(1000, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw)

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

        for i in range(0, num_samples):
            sample = samples[i]
            t = timestamp[i]
            for name, value in zip(CHANNEL_NAMES, sample):
                self.data[name].append(value)
            self.timestamps.append(t)

        # Filter only the new samples from this update (causal, using
        # persistent filter state), then append the results to
        # filtered_data instead of re-filtering the whole buffer.
        for name in CHANNEL_NAMES:
            new_samples = np.array([samples[i][CHANNEL_NAMES.index(name)] for i in range(num_samples)])
            filtered_chunk, self.filt_zi[name] = sps.lfilter(
                self.filt_b, self.filt_a, new_samples, zi=self.filt_zi[name]
            )
            self.filtered_data[name].extend(filtered_chunk)

        # redraw each curve with the latest filtered buffer contents
        for name in CHANNEL_NAMES:
            if len(self.filtered_data[name]) > 100:
                self.curves[name].setData(list(self.timestamps), list(self.filtered_data[name]))


if __name__ == "__main__":
    main()