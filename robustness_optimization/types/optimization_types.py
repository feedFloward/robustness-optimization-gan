from robustness_optimization.types.helpers import uniform, scale_to_param_range, make_discrete, split_up_sampling, normalize, type_casting

from typing import List

def _to_dict(func):
    def wrapper_to_dict(config, sampling_model=None, *args, **kwargs):
        sample = func(config, sampling_model, *args, **kwargs)
        sample = split_up_sampling(sample, config.__dict__.items())
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
            value = scale_to_param_range(value, param_setting)

            #discretize
            if param_setting.discrete:
                value = make_discrete(value)

            #normalize
            if param_setting.mixture:
                value = normalize(value)

            value = type_casting(value, param_setting)

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
        return uniform(self._output_dim())

    @_transform
    @_to_dict
    def from_model(self, sampling_model, *args, **kwargs):
        return sampling_model.generate_samples(1)


class Design:
    '''
    sollte können:
        methoden um configs zu sortieren anhand von SN Ratio
    '''
    def __init__(self, configurations : List[Configuration]):
        self.state = configurations

    def sortblabla(self):
        pass


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

class Optimization:
    '''
    contains methods for main logic
    '''
    def __init__(self, settings, simulation_model, factor_sampling_model, noise_sampling_model):
        self.settings = settings
        self.simulation_model = simulation_model
        self.factor_design_maker = DesignMaker(parameter= settings.factor_definition, sampling_model= factor_sampling_model, **settings.factor_design_definition())
        self.noise_design_maker = DesignMaker(parameter= settings.noise_definition, sampling_model= noise_sampling_model, **settings.noise_design_definition())

        self.factor_candidates = self.factor_design_maker.get_uniform_sample()
        self.noise_candidates = self.noise_design_maker.get_uniform_sample()