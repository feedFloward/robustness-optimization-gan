from robustness_optimization.types import helpers, visualization_helpers

from typing import List, Dict


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

            # discretize
            if param_setting.discrete:
                value = helpers.make_discrete(value)

            # normalize
            if param_setting.mixture:
                value = helpers.normalize(value)

            value = helpers.type_casting(value, param_setting)

            sample_dict.update({param : value})

        return sample_dict
    return wrapper_transform


def _re_transform(func):
    def wrapper_re_transform(maker_object, feedback : List[Dict], *args, **kwargs):
        sample_to_array = helpers.retransform(feedback, maker_object)
        return func(maker_object, feedback= sample_to_array, *args, **kwargs)
    return wrapper_re_transform


def config_unique_in_design(func):
    def wrapper(maker_object, curr_champion=None, *args, **kwargs):
        design_to_build = []
        num_attempts_to_create = 0
        if curr_champion:
            design_to_build.append(curr_champion)
        while len(design_to_build) < maker_object.num_samples:
            if num_attempts_to_create == 10: # 10 ist willkürlich
                return None
            # check if configuration is unique
            config = func(maker_object)
            if not helpers.check_if_config_in_design(config, design_to_build):
                design_to_build.append(config)
            else:
                num_attempts_to_create += 1
        
        # return Design([Configuration(**self.parameter.__dict__).from_uniform() for sample in range(self.num_samples)])
        return Design(design_to_build)
    return wrapper


class Configuration:
    def __init__(self, **kwargs):
        """
        Argumente müssten Vektor von Werten sein
        Zuordnung zu den namen der Faktoren über irgendein mapping (indizes)

        sollte können:
            funktion um für configuration SN Ration berechnen zu können
        """
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
        return sampling_model.generate_samples()

    @_transform
    @_to_dict
    def from_array(self, values):
        return values


class Design:
    """
    sollte können:
        methoden um configs zu sortieren anhand von SN Ratio
    """
    def __init__(self, configurations : List[Configuration]):
        self.state = configurations

    def __iter__(self):
        return iter(self.state)

    def __next__(self):
        return self.state

    def __getitem__(self, index):
        return self.state[index]

    def get_best_configuration(self, sn_calc_func):
        for config in self.state:
            config.update({'sn_ratio': sn_calc_func(config['response'])})

        # evtl. muss unabhängig vom optimierungsziel immer das erste element gewählt werden?
        if sn_calc_func.__name__ == 'larger_the_better':
            return sorted(self.state, key= lambda conf : conf['sn_ratio'])[-1]
        elif sn_calc_func.__name__ == 'smaller_the_better':
            return sorted(self.state, key= lambda conf : conf['sn_ratio'])[0]

    def plot_design(self):
        return visualization_helpers.plot_design(self.state)


class NoiseDesign:
    def __init__(self, configurations : List[Configuration]):
        self.state = configurations
        self.robustness = None

    def calc_robustness(self, sn_calc_func):
        self.robustness = sn_calc_func([noise_conf['response'] for noise_conf in self.state])

    def __iter__(self):
        return iter(self.state)

    def __next__(self):
        return self.state

    def __getitem__(self, index):
        return self.state[index]


class DesignList:
    """
    list of noise designs
    """
    def __init__(self, designs):
        self.state = designs

    def __iter__(self):
        return iter(self.state)

    def __next__(self):
        return self.state

    def __getitem__(self, index):
        return self.state[index]
    
    def get_worst_design(self, sn_calc_func):
        # for design in self.state:
            # design.append({'sn_ratio': sn_calc_func([noise_conf['response'] for noise_conf in design])})
        for design in self.state:
            design.calc_robustness(sn_calc_func)
        return sorted(self, key= lambda design: design.robustness)[0]
        # return sorted(self.state, key= lambda design : next(filter(lambda item: 'sn_ratio' in item.keys(), design))['sn_ratio'])[0]


class DesignMaker:
    """
    sollte können:
        anhand von Settings.factor_definition/noise_definition initialisiert werden
        methoden zum samplen haben (mit interface zum gan)
        methoden zum normalisiern/skalieren haben
        ...
    modell im hintergrund braucht die methoden:
        generate_samples
        update
    """
    def __init__(self, num_samples, parameter, sampling_model):
        self.num_samples = num_samples
        self.parameter = parameter
        self.sampling_model = sampling_model

    @config_unique_in_design
    def get_uniform_sample(self):
        return Configuration(**self.parameter.__dict__).from_uniform()

    @config_unique_in_design
    def get_sample_from_model(self):
        # return Design([Configuration(**self.parameter.__dict__).from_model(self.sampling_model) for sample in range(self.num_samples)])
        return Configuration(**self.parameter.__dict__).from_model(self.sampling_model)

    @_re_transform
    def update_model(self, feedback):
        self.sampling_model.update(feedback)


class NoiseDesignMaker(DesignMaker):
    def __init__(self, num_noise_designs, **kwargs):
        super(NoiseDesignMaker, self).__init__(**kwargs)
        self.num_noise_designs = num_noise_designs

    def get_uniform_sample(self):
        return DesignList([NoiseDesign([Configuration(**self.parameter.__dict__).from_uniform() for sample in range(self.num_samples)]) for design in range(self.num_noise_designs)])
        # return DesignList([[Configuration(**self.parameter.__dict__).from_uniform() for sample in range(self.num_samples)] for design in range(self.num_noise_designs)])

    def get_sample_from_model(self):
        # sample = self.sampling_model.generate_samples()
        # sample = helpers.reshape_to_design(sample, self.num_samples)
        # return DesignList([NoiseDesign([Configuration(**self.parameter.__dict__).from_array(config) for config in sample]) for design in range(self.num_noise_designs)])
        return DesignList([NoiseDesign([Configuration(**self.parameter.__dict__)\
        .from_array(config) for config in helpers.reshape_to_design(self.sampling_model.generate_samples(), self.num_samples)])\
         for design in range(self.num_noise_designs)])
        # return DesignList([[Configuration(**self.parameter.__dict__).from_array(config) for config in sample] for design in range(self.num_noise_designs)])
