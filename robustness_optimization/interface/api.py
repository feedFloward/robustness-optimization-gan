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
        return self.model.simulation_run(factor_config= set_factors(**factor_config), 
                                         noise_config= set_noise(**noise_config),
                                        )

    def main(self, factor_design, noise_design, competitor_flag : str):
        '''
        performs one complete iteration of GAN optimization
        
        Keyword arguments:
        factor_design -- list of factor configurations
        noise_design -- list of noise configurations
        competitor_flag -- 'factor' or 'noise' defines whether factor gan or noise gan is testing against the current oposite champion
        
        Returns:
        updated dict with key 'target' and list of target values
        '''
        
        if competitor_flag == 'factor':
            for factor_config in factor_design:
                target_vals = []
                for noise_config in noise_design:
                    target_vals.append(self.run(factor_config= factor_config, noise_config= noise_config))
                factor_config.update({'target': target_vals})
            return factor_design
        elif competitor_flag == 'noise':
            for noise_plan in noise_design:
                target_vals = []
                for noise_config in noise_plan:
                    target_vals.append(self.run(factor_config= factor_design[0], noise_config= noise_config))
                noise_plan.append({'target': target_vals})
            return noise_design