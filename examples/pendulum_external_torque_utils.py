import numpy as np

#
# Micro-implementation of the Neural Engineering Framework
#

class ReLU:
    @staticmethod
    def inverse(a):
        return a

    @staticmethod
    def activity(x):
        return np.maximum(0, x)

class LIF:
    slope = 2.0 / 3.0

    @staticmethod
    def inverse(a):
        valid = a > 0
        return 1.0 / (1.0 - np.exp(LIF.slope - (1.0 / (valid * a + 1e-6))))

    @staticmethod
    def activity(x):
        valid = x > (1.0 + 1e-6)
        return valid / (LIF.slope - np.log(1.0 - valid * (1.0 / x)))


class Ensemble:
    def __init__(self, n_neurons, n_dimensions, neuron_type=LIF):
        self.neuron_type = neuron_type

        # Randomly select the intercepts and the maximum rates
        self.intercepts = np.random.uniform(-0.95, 0.95, n_neurons)
        self.max_rates = np.random.uniform(0.5, 1.0, n_neurons)

        # Randomly select the encoders
        self.encoders = np.random.normal(0, 1, (n_neurons, n_dimensions))
        self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, None]

        # Compute the current causing the maximum rate/the intercept
        J_0 = self.neuron_type.inverse(0)
        J_max_rates = self.neuron_type.inverse(self.max_rates)

        # Compute the gain and bias
        self.gain = (J_0 - J_max_rates) / (self.intercepts - 1.0)
        self.bias = J_max_rates - self.gain

    def __call__(self, x):
        return self.neuron_type.activity(self.J(x))

    def J(self, x):
        return self.gain[:, None] * self.encoders @ x + self.bias[:, None]

#
# Delay Network Stuff (see Voelker, Eliasmith 2018; Neural Computation)
#

def make_delay_network(q, theta):
    Q = np.arange(q, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i - j + 1)) * R
    B = (-1.)**Q[:, None] * R
    return A, B


def discretize_lti(dt, A, B):
    import scipy.linalg
    Ad = scipy.linalg.expm(A * dt)
    Bd = np.dot(np.dot(np.linalg.inv(A), (Ad - np.eye(A.shape[0]))), B)
    return Ad, Bd