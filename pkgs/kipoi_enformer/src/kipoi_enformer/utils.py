import tensorflow as tf


class RandomModel(tf.keras.Model):
    """
    A random model for testing purposes.
    """

    def __init__(self, lamda=10):
        super().__init__()
        self.lamda = lamda

    def predict_on_batch(self, input_tensor):
        # tf.random.set_seed(42)
        return {
            'human': tf.abs(tf.random.poisson((input_tensor.shape[0], 896, 5313,), lam=self.lamda)),
            'mouse': tf.abs(tf.random.poisson((input_tensor.shape[0], 896, 1643), lam=self.lamda)),
        }
