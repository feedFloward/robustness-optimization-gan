from robustness_optimization.name_tbd_initialization import read_settings
from robustness_optimization.types.optimization_types import DesignMaker
from robustness_optimization.gan.vanilla_gan import GAN
from robustness_optimization import name_tbd_initialization
import os

def main():
    settings = read_settings()

    factor_sampling_model = GAN(**settings.factor_gan_parameter())
    noise_sampling_model = GAN(**settings.noise_gan_parameter())

    factor_design_maker = DesignMaker(parameter= settings.factor_definition, sampling_model= factor_sampling_model, **settings.factor_design_definition())
    noise_design_maker = DesignMaker(parameter= settings.noise_definition, sampling_model= noise_sampling_model, **settings.noise_design_definition())

    print("Uniform Sampling for factors: ")
    print(factor_design_maker.get_uniform_sample().state)

    print("GAN Sampling for factors: ")
    print(factor_design_maker.get_sample_from_model().state)

    print("Uniform Sampling for noise: ")
    print(noise_design_maker.get_uniform_sample().state)

    print("GAN Sampling for noise: ")
    print(noise_design_maker.get_sample_from_model().state)

def test_settings_variation():
    various_settings = os.listdir(r"./settings/")

    for setting in various_settings:
        print(f"test with {setting}")
        name_tbd_initialization.SETTINGS_PATH = r"./settings/" + setting

        main()