from robustness_optimization.name_tbd_initialization import read_settings
from robustness_optimization.types.optimization_types import Optimization
from robustness_optimization.gan.vanilla_gan import GAN
from robustness_optimization.interface import SimpyModel

def main():
    settings = read_settings()

    factor_sampling_model = GAN(**settings.factor_gan_parameter())
    noise_sampling_model = GAN(**settings.noise_gan_parameter())

    simulation_model = SimpyModel("C:/Users/fconrad/git/robustness-optimization-gan/simpy_case_study/model.py")

    optimization = Optimization(
        settings= settings,
        simulation_model= simulation_model,
        factor_sampling_model= factor_sampling_model,
        noise_sampling_model= noise_sampling_model,
    )

    optimization.run()