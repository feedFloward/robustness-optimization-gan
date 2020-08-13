import argparse
from robustness_optimization.types.sim_types import ParameterConfig, NoiseConfig

def set_factors(**kwargs):
    return ParameterConfig(**kwargs)

def set_noise(**kwargs):
    return NoiseConfig(**kwargs)

class ModelInterface:
    
    def run(self, factor_config, noise_config):
        '''
        Performs one single simulation run with parameters and noise set in...
        '''
        self.model.simulation_run(factor_config= set_factors(**factor_config),
                                  noise_config= set_noise(**noise_config),
                                  )


    def main(self, factor_design, noise_design, competitor_flag : str):
        '''
        performs one complete iteration of GAN optimization
        
        Keyword arguments:
        factor_design -- list of factor configurations
        noise_design -- list of noise configurations
        competitor_flag -- 'factor' or 'noise' defines whether factor gan or noise gan is testing against the current oposite champion
        '''
        if competitor_flag == 'factor':
            for config in factor_design: self.run(factor_config= config, noise_config= noise_design[0])
        elif competitor_flag == 'noise':
            for config in noise_design: self.run(factor_config= factor_design[0], noise_config= config)
        pass

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-factor_dict", help="path to json with factor name and value as key value pairs", type=str)
#     parser.add_argument("-noise_dict", type=str)
#     parser.add_argument("-simulation_settings", type=str)
#     args = parser.parse_args()
#     print(args.factor_dict)
#     print(args.noise_dict)
#     print(args.simulation_settings)
#     '''
#     json files auslesen und main mit argumenten aufrufen...
#     '''
#     print('...ended')
#     main(args)