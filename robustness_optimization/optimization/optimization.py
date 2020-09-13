from robustness_optimization.types.optimization_types import DesignMaker, NoiseDesignMaker
from robustness_optimization.types import helpers

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

class Optimization:
    '''
    contains methods for main logic
    '''
    def __init__(self, settings, simulation_model, factor_sampling_model, noise_sampling_model, out_path= None):
        self.out_path = out_path
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

        self.sn_history = []

    def run(self):
        print("find best candidate from initial factor design...\n")
        self.factor_champion = self.evaluate_factor_design()
        #assign initial sn ratio to initial noise design:
        self.noise_champion.robustness = self.factor_champion['sn_ratio']
        print("best configuration found:")
        print(self.factor_champion)
        self.sn_ratio = self.factor_champion['sn_ratio']
        self.sn_history.append(self.sn_ratio)
        # round sn_ratio -> evtl. später als funktion
        print("current SN Ratio:")
        print(self.sn_ratio)

        while self.attempts < 3:
            self.attempts += 1
            self.iteration_factors()
            self.iteration_noise()

        self.print_results()

        return {
            "best_factor_config": self.factor_champion,
            "worst_noise_design": self.noise_champion,
            "sn_history": self.sn_history
        }


    @print_iteration
    def iteration_factors(self):
        self.iterations += 1
        self.factor_candidates = self.factor_design_maker.get_sample_from_model(curr_champion= self.factor_champion)
        
        #if no new configurations were generated:
        if self.factor_candidates == None:
            print("WARNING! sampling model was not able to create a design of unique configurations after 10 attempts!")
            return
        
        # if self.out_path:
        #     self.factor_candidates.plot_design().savefig(self.out_path+'factor_output_iteration_'+str(self.iterations)+'.png')
        print("factor design tested:")
        self._print_design(self.factor_candidates)
        contender = self.evaluate_factor_design()
        print("best factor configuration evaluated:")
        print(contender)
        print("with sn_ratio:")
        print(contender['sn_ratio'])

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
        self.sn_history.append(self.sn_ratio)

    @print_iteration
    def iteration_noise(self):
        self.iterations += 1
        self.noise_candidates = self.noise_design_maker.get_sample_from_model()
        # append current noise champion (always tested with changed factor config)
        # self.noise_candidates.state.append(self.noise_champion)
        contender = self.evaluate_noise_designs()
        print("worst noise design evaluated")
        self._print_design(contender)
        print("with sn ratio:")
        print(contender.robustness)
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
        self.sn_history.append(self.sn_ratio)



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