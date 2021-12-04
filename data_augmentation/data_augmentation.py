import math
import numpy as np
from scipy.signal import butter,filtfilt

class DataAugmentation():

    def __init__(self, audio):

        self.audio = audio

        self.fs = 44.1
        self.lp_cutoff = 3
        self.hp_cutoff = 2
        self.order = 2

    def apply_gaussian_noise(self):
        # noise = np.random.randn(*features.shape).astype(np.float32)

        dimension = self.audio.shape[0]
        RMS = math.sqrt(np.mean(self.audio**2))
        noise = np.random.normal(0, RMS / 2, dimension)

        randomness = np.random.randint(2, size=dimension)
        noise = randomness * noise

        samples = self.audio + noise
        return samples

    def apply_lp(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.lp_cutoff / nyq
        # Get the filter coefficients
        b, a = butter(self.order, normal_cutoff, btype='lowpass', analog=False)
        y = filtfilt(b, a, self.audio)
        return y

    def apply_hp(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.hp_cutoff / nyq
        # Get the filter coefficients
        b, a = butter(self.order, normal_cutoff, btype='highpass', analog=False)
        y = filtfilt(b, a, self.audio)
        return y

