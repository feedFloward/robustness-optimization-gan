from robustness_optimization.types.helpers import uniform

from typing import List

def _to_dict(func):
    def wrapper_to_dict(config, sampling_model=None):
        sample = func(config, sampling_model)
        current_config = {}
        for (param_name, value) in zip(config.__dict__.keys(), sample):
            current_config.update({param_name: value})
        return current_config
    return wrapper_to_dict


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
        self.num_parameter = len(self.__dict__)

        
    @_to_dict
    def from_uniform(self, *args, **kwargs):
        return uniform(self.num_parameter)

    @_to_dict
    def from_model(self, sampling_model):
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
    '''
    def __init__(self, num_samples, parameter, sampling_model):
        self.num_samples = num_samples
        self.parameter = parameter
        self.sampling_model = sampling_model

    def get_uniform_sample(self):
        return Design([Configuration(**self.parameter.__dict__).from_uniform() for sample in range(self.num_samples)])

    def get_sample_from_model(self):
        return Design([Configuration(**self.parameter.__dict__).from_model(self.sampling_model) for sample in range(self.num_samples)])

