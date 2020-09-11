from robustness_optimization.name_tbd_initialization import read_settings
from robustness_optimization.optimization import Optimization
from robustness_optimization.gan.vanilla_gan import GAN
from robustness_optimization.interface import SimpyModel
from robustness_optimization.types.sampling_model import SamplingModel
from robustness_optimization.types import output

import argparse
import sys

# @output.write_to_output
def main(sim_model_type, sim_model_path, sampling_model_type):
    sys.stdout = output.DualOutput()


    settings = read_settings()

    factor_sampling_model = SamplingModel(**settings.factor_gan_parameter(), model_flag= sampling_model_type)
    noise_sampling_model = SamplingModel(**settings.noise_gan_parameter(), model_flag= sampling_model_type)

    # simulation_model = SimpyModel("C:/Users/fconrad/git/robustness-optimization-gan/simpy_case_study/model.py")
    if sim_model_type == 'simpy':
        simulation_model = SimpyModel(sim_model_path)

    optimization = Optimization(
        settings= settings,
        simulation_model= simulation_model,
        factor_sampling_model= factor_sampling_model,
        noise_sampling_model= noise_sampling_model,
    )

    history = optimization.run()

    output.write_results(history)


parser = argparse.ArgumentParser()
parser.add_argument('--sim-model-type', help="choose between \'simpy\', \'plant-simulation\'")
parser.add_argument('--sim-model-path')
parser.add_argument('--sampling-model-type')

args = parser.parse_args()

if __name__ == "__main__":
    print(args.sim_model_path)
    main(args.sim_model_type, args.sim_model_path, args.sampling_model_type)