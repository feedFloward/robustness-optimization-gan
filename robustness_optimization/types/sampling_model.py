from robustness_optimization.gan import GAN
from robustness_optimization.types.helpers import flatten
import numpy as np


def switch_backend_model(model_flag):
    switcher = {
        'vanilla-gan': GAN
    }

    return switcher.get(model_flag, "model unknown")


class SamplingModel:
    def __init__(self, model_flag, **kwargs):
        model = switch_backend_model(model_flag)
        self.sampling_model = model(**kwargs)
        self.sampling_model.compile()
        # intializize model_output to range -1, 1
        self.update()

    def generate_samples(self):
        return self.sampling_model.generate_samples(1)

    def update(self, feedback=None):
        """
        wrapper for training model with feedback (best/worst factor/noise)
        """
        # flatten feedback (in case of noise design)
        if feedback:
            feedback = flatten(feedback)
            train_database = np.repeat(feedback, 100, axis=0)
            # add "noisy" configurations
            noise_added = np.array([[np.random.normal(val, scale=0.4) for val in feedback[0]] for i in range(100)])
            noise_added = np.clip(noise_added, -1, 1)
            train_database = np.concatenate([train_database, noise_added], axis=0)
        else:
            # no feedback when sampling model is initialized
            print("INITIAL GAN TRAINING...")
            train_database = np.random.uniform(-1, 1, size=(1000, self.sampling_model.generator.output_dim))

        self.sampling_model.fit(train_database, epochs=3, verbose=1)
