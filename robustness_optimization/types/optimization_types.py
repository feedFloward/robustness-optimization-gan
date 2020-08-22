from robustness_optimization.types import helpers

from typing import List

def _to_dict(func):
    def wrapper_to_dict(config, sampling_model=None, *args, **kwargs):
        sample = func(config, sampling_model, *args, **kwargs)
        sample = helpers.split_up_sampling(sample, config.__dict__.items())
        current_config = {}
        for (param_name, value) in zip(config.__dict__.keys(), sample):
            current_config.update({param_name: value})
        return current_config
    return wrapper_to_dict

def _transform(func):
    def wrapper_transform(config, *args, **kwargs):
        sample_dict = func(config, *args, **kwargs)
        
        for (param, value) in sample_dict.items():
            # scales each parameter to its range:
            param_setting = config.__dict__[param]
            value = helpers.scale_to_param_range(value, param_setting)

            #discretize
            if param_setting.discrete:
                value = helpers.make_discrete(value)

            #normalize
            if param_setting.mixture:
                value = helpers.normalize(value)

            value = helpers.type_casting(value, param_setting)

            sample_dict.update({param : value})


        return sample_dict
    return wrapper_transform


class Configuration:
    def __init__(self, **kwargs):
        '''
        Argumente müssten Vektor von Werten sein
        Zuordnung zu den namen der Faktoren über irgendein mapping (indizes)

        sollte können:
            funktion um für configuration SN Ration berechnen zu können
        '''
        for (key, val) in kwargs.items():
            self.__dict__.update({key: val})

    def _output_dim(self):
        return sum([param.num_mixture_components if param.mixture else 1 for (name, param) in self.__dict__.items()])

    @_transform
    @_to_dict
    def from_uniform(self, *args, **kwargs):
        return helpers.uniform(self._output_dim())

    @_transform
    @_to_dict
    def from_model(self, sampling_model, *args, **kwargs):
        return sampling_model.generate_samples(1)

    @_transform
    @_to_dict
    def from_array(self, values):
        return values


class Design:
    '''
    sollte können:
        methoden um configs zu sortieren anhand von SN Ratio
    '''
    def __init__(self, configurations : List[Configuration]):
        self.state = configurations

    def get_best_configuration(self, sn_calc_func):
        for config in self.state:
            config.update({'sn_ratio': sn_calc_func(config['response'])})
        if sn_calc_func.__name__ == 'larger_the_better':
            return sorted(self.state, key= lambda conf : conf['sn_ratio'])[-1]
        elif sn_calc_func.__name__ == 'smaller_the_better':
            return sorted(self.state, key= lambda conf : conf['sn_ratio'])[0]

class NoiseDesign:
    '''
    list of noise designs
    '''
    def __init__(self, designs):
        self.state = designs

class DesignMaker:
    '''
    sollte können:
        anhand von Settings.factor_definition/noise_definition initialisiert werden
        methoden zum samplen haben (mit interface zum gan)
        methoden zum normalisiern/skalieren haben
        ...
    modell im hintergrund braucht die methoden:
        generate_samples
        update
    '''
    def __init__(self, num_samples, parameter, sampling_model):
        self.num_samples = num_samples
        self.parameter = parameter
        self.sampling_model = sampling_model

    def get_uniform_sample(self):
        return Design([Configuration(**self.parameter.__dict__).from_uniform() for sample in range(self.num_samples)])

    def get_sample_from_model(self):
        return Design([Configuration(**self.parameter.__dict__).from_model(self.sampling_model) for sample in range(self.num_samples)])

    def update_model(self, design):
        self.sampling_model.update(design)


class NoiseDesignMaker(DesignMaker):
    def __init__(self, num_noise_designs, **kwargs):
        super(NoiseDesignMaker, self).__init__(**kwargs)
        self.num_noise_designs = num_noise_designs

    def get_uniform_sample(self):
        return NoiseDesign([[Configuration(**self.parameter.__dict__).from_uniform() for sample in range(self.num_samples)] for design in range(self.num_noise_designs)])

    def get_sample_from_model(self):
        sample = self.sampling_model.generate_samples(1)
        sample = helpers.reshape_to_design(sample, self.num_samples)
        return NoiseDesign([[Configuration(**self.parameter.__dict__).from_array(config) for config in sample] for design in range(self.num_noise_designs)])


class Optimization:
    '''
    contains methods for main logic
    '''
    def __init__(self, settings, simulation_model, factor_sampling_model, noise_sampling_model):
        self.settings = settings
        self.simulation_model = simulation_model
        self.factor_design_maker = DesignMaker(parameter= settings.factor_definition, sampling_model= factor_sampling_model, **settings.factor_design_definition())
        self.noise_design_maker = NoiseDesignMaker(parameter= settings.noise_definition, sampling_model= noise_sampling_model, **settings.noise_design_definition())

        self.factor_candidates = self.factor_design_maker.get_uniform_sample()
        self.noise_candidates = self.noise_design_maker.get_uniform_sample()

        #set noise champion for initialization randomly:
        self.noise_champion = self.noise_candidates.state[0]

        self.sn_calc_func = helpers.get_sn_calc_func(**self.settings.response_definition())

        self.attempts = 0
        self.iterations = 0

    def run(self):
        print("find best candidate from initial factor design...\n")
        self.factor_champion = self.evaluate_factor_design()
        print("best configuration found:")
        print(self.factor_champion)

        while self.attempts < 3:
            self.attempts += 1
            self.iteration_factors()

    def iteration_factors(self):
        self.iterations += 1
        print("~~~~~~~~~~~~~~~~\n")
        print(f"iteration #{self.iterations} - FACTORS")
        print("best factor configuration so far: ")
        print(self.factor_champion)
        self.factor_candidates = self.factor_design_maker.get_uniform_sample()
        print("factor design tested:")
        self._print_design(self.factor_candidates.state)
        contender = self.evaluate_factor_design()
        print("best factor configuration evaluated:")
        print(contender)
        if contender["sn_ratio"] > self.factor_champion["sn_ratio"]:
            print("new champion found")
            self.factor_champion = contender
            self.attempts = 0
        else:
            print(f"no improvement after {self.attempts} attempts...")

    def iteration_noise(self):
        self.iterations += 1
        print("~~~~~~~~~~~~~~~~\n")
        print(f"iteration #{self.iterations} - NOISE")
        print("worst noise design so far:")
        print(self.noise_champion)
        self.noise_candidates = self.noise_design_maker.get_uniform_sample()



    def evaluate_factor_design(self):
        print("evaluate factor design:\n")
        self._print_design(self.factor_candidates.state)
        print("with noise design:\n")
        self._print_design(self.noise_champion)
        self.factor_candidates.state = self.simulation_model.main(factor_design= self.factor_candidates.state, noise_design= self.noise_champion, competitor_flag= 'factor')
        return self.factor_candidates.get_best_configuration(self.sn_calc_func)

    def evaluate_noise_designs(self):
        print("evaluate noise designs:\n")
        for design in self.noise_candidates.state:
            self._print_design(design)
        print("on champion factor config:")
        print(self.factor_champion)
        print(self.simulation_model.main(factor_design= self.factor_champion, noise_design= self.noise_candidates.state, competitor_flag= 'noise'))



    def _print_design(self, design):
        print(40*"_")
        for config in design:
            print("|" + str(config) + "|")
        print(40*"_")