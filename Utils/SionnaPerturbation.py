import random
import numpy as np

class SionnaPerturbation:

    def __init__(self, perturb_data):

        self.perturb_data = perturb_data

        self.factor_mean = np.mean(
            [x["factor"] for x in perturb_data]
        )

    def __call__(self, csi):

        sample = random.choice(
            self.perturb_data
        )

        factor = (
            sample["factor"]
            / self.factor_mean
        )

        csi = csi.copy()

        csi[0] *= factor

        return csi