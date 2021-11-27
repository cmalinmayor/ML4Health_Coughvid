import math
import numpy as np
from scipy.signal import butter,filtfilt

class DataAugmentation():

    def __init__(self,
                 audio,
                 apply_gaussian_noise=True,
                 apply_lp=False,
                 apply_hp=True):

        self.audio = audio
        self.apply_gaussian_noise = apply_gaussian_noise
        self.apply_lp = apply_lp
        self.apply_hp = apply_hp
        # TODO: still need to figure out how to choose the filter requirements.
        self.fs = 30.0  # sample rate, Hz
        self.cutoff = 3  # desired cutoff frequency of the filter, Hz (the higher cutoff, the less effects applied for lowpass
        self.order = 2  # sin wave can be approx represented as quadratic

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
        normal_cutoff = self.cutoff / nyq
        # Get the filter coefficients
        b, a = butter(self.order, normal_cutoff, btype='lowpass', analog=False)
        y = filtfilt(b, a, self.audio)
        return y

    def apply_hp(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        # Get the filter coefficients
        b, a = butter(self.order, normal_cutoff, btype='highpass', analog=False)
        y = filtfilt(b, a, self.audio)
        return y

