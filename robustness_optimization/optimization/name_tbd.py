from robustness_optimization.name_tbd_initialization import read_settings
from robustness_optimization.types.optimization_types import DesignMaker
from robustness_optimization.gan.vanilla_gan import GAN

def main():
    settings = read_settings()
    factor_design_maker = DesignMaker(parameter= settings.factor_definition, sampling_model= GAN(**settings.factor_gan_parameter()),**settings.factor_design_definition())
    initial_factor_design = factor_design_maker.get_uniform_sample()
    print(initial_factor_design.state)

    noise_design_maker = DesignMaker(parameter= settings.noise_definition, sampling_model= GAN(**settings.noise_gan_parameter()), **settings.noise_design_definition())
    initial_noise_design = noise_design_maker.get_uniform_sample()
    print(initial_noise_design.state)