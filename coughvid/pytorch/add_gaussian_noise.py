import numpy as np

class AddGaussianNoise():
    """Add gaussian noise to the samples"""

    def __init__(self, sigma=1):

        self.sigma = sigma

    def apply(self, features):
        #noise = np.random.randn(*features.shape).astype(np.float32)

        row, col = features.shape[0], features.shape[1]
        noise = np.random.normal(0, self.sigma, (row, col))

        randomness = np.random.randint(2, size=(row, col))
        noise = randomness * noise

        samples = features + noise
        return samples
