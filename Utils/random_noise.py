import numpy as np
# CSI데이터에 세기왜곡을 넣어줌 
def add_gaussian_noise(csi, sigma=0.03):

    noise = np.random.normal(
        0,
        sigma,
        size=csi.shape
    )

    return csi + noise

# CSI데이터에 위상 왜곡을 넣어줌 
def add_phase_noise(csi, alpha=0.1):

    amplitude = np.abs(csi)
    phase = np.angle(csi)

    delta = np.random.uniform(
        -alpha,
        alpha,
        size=phase.shape
    )

    new_phase = phase + delta

    perturbed = amplitude * np.exp(1j * new_phase)

    return perturbed

# noise 클래스 - Gaussian 노이즈와 위상 노이즈를 랜덤하게 적용하는 클래스
class RandomNoisePerturbation:

    def __init__(
        self,
        gaussian_prob=0.5,
        phase_prob=0.5,
        sigma=0.03,
        alpha=0.1
    ):

        self.gaussian_prob = gaussian_prob
        self.phase_prob = phase_prob

        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, csi):

        x = csi.copy()

        if np.random.rand() < self.gaussian_prob:
            x = add_gaussian_noise(
                x,
                self.sigma
            )

        if np.random.rand() < self.phase_prob:
            x = add_phase_noise(
                x,
                self.alpha
            )

        return x