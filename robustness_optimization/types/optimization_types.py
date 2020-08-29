from robustness_optimization.types import helpers

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
            if num_attempts_to_create == 10: #10 ist willkürlich
                return None
            #check if configuration is unique
            config = func(maker_object)
            if not helpers.check_if_config_in_design(config, design_to_build):
                design_to_build.append(config)
            else:
                num_attempts_to_create += 1
        
        # return Design([Configuration(**self.parameter.__dict__).from_uniform() for sample in range(self.num_samples)])
        return Design(design_to_build)
    return wrapper


def print_iteration(func):
    def print_wrapper(optimization, *args, **kwargs):
        print(40*"=")
        print(f"ITERATION {optimization.iterations} === {func.__name__}")
        print(40*"=")
        print(f"current SN Ratio:\t{optimization.sn_ratio}")
        print("\n")
        print("best factor configuration: ")
        print(optimization.factor_champion)
        print("\n")
        print("worst noise design: ")
        optimization._print_design(optimization.noise_champion)
        func(optimization)
        print(f"current SN Ratio after iteration:\t{optimization.sn_ratio}")
        
    return print_wrapper


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
        return sampling_model.generate_samples()

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
    '''
    list of noise designs
    '''
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
        self.noise_champion = self.noise_candidates[0]

        self.sn_calc_func = helpers.get_sn_calc_func(**self.settings.response_definition())

        self.attempts = 0
        self.iterations = 0

    def run(self):
        print("find best candidate from initial factor design...\n")
        self.factor_champion = self.evaluate_factor_design()
        #assign initial sn ratio to initial noise design:
        self.noise_champion.robustness = self.factor_champion['sn_ratio']
        print("best configuration found:")
        print(self.factor_champion)
        self.sn_ratio = self.factor_champion['sn_ratio']
        # round sn_ratio -> evtl. später als funktion
        print("current SN Ratio:")
        print(self.sn_ratio)

        while self.attempts < 3:
            self.attempts += 1
            self.iteration_factors()
            self.iteration_noise()

        self.print_results()


    @print_iteration
    def iteration_factors(self):
        self.iterations += 1
        self.factor_candidates = self.factor_design_maker.get_sample_from_model(curr_champion= self.factor_champion)
        
        #if no new configurations were generated:
        if self.factor_candidates == None:
            print("WARNING! sampling model was not able to create a design of unique configurations after 10 attempts!")
            return
        #append current factor champion (champions always tested with changed noise design!)
        # self.factor_candidates.state.append(self.factor_champion)
        print("factor design tested:")
        self._print_design(self.factor_candidates)
        contender = self.evaluate_factor_design()
        print("best factor configuration evaluated:")
        print(contender)

        #DIESER ABGLEICH WIRD ERSETZT DURCH: IST CONTENDER GLEICH DEM ALTEN CHAMP???????
        if contender["sn_ratio"] > self.sn_ratio:
            print("new champion found:")
            print(contender)
            print("with sn-ratio: ")
            print(contender["sn_ratio"])
            self.factor_champion = contender
            self.factor_design_maker.update_model(self.factor_champion)
            self.sn_ratio = contender["sn_ratio"]
            self.attempts = 0
        else:
            print(f"no improvement after {self.attempts} attempts...")

    @print_iteration
    def iteration_noise(self):
        self.iterations += 1
        self.noise_candidates = self.noise_design_maker.get_sample_from_model()
        # append current noise champion (always tested with changed factor config)
        # self.noise_candidates.state.append(self.noise_champion)
        contender = self.evaluate_noise_designs()
        print("worst noise design evaluated")
        self._print_design(contender)
        if contender.robustness < self.sn_ratio:
            print("new noise champion found")
            self._print_design(contender)
            print("with sn-ratio: ")
            print(contender.robustness)
            self.noise_champion = contender
            self.noise_design_maker.update_model(self.noise_champion)
            self.sn_ratio = contender.robustness
            self.attempts = 0
        else:
            print(f"no decline in sn ratio after {self.attempts} attempts...")



    def evaluate_factor_design(self):
        print("evaluate factor design:\n")
        self._print_design(self.factor_candidates)
        print("with noise design:\n")
        self._print_design(self.noise_champion)
        self.factor_candidates = self.simulation_model.main(factor_design= self.factor_candidates, noise_design= self.noise_champion, competitor_flag= 'factor')
        return self.factor_candidates.get_best_configuration(self.sn_calc_func)

    def evaluate_noise_designs(self):
        print("evaluate noise designs:\n")
        for design in self.noise_candidates:
            self._print_design(design)
        print("on champion factor config:")
        print(self.factor_champion)
        self.noise_candidates = self.simulation_model.main(factor_design= self.factor_champion, noise_design= self.noise_candidates, competitor_flag= 'noise')
        return self.noise_candidates.get_worst_design(self.sn_calc_func)


    def _print_design(self, design):
        print(40*"_")
        for config in design:
            print("|" + str(config) + "|")
        print(40*"_")

    def print_results(self):
        print("finished optimization with best factor configuration:")
        print(self.factor_champion)
        print("and noise design:")
        self._print_design(self.noise_champion)