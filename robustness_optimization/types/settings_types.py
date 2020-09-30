from typing import Dict, List


# Wording 'Parameter' nochmal überdenken (evtl. 'Definition' oder so)
# !!!! Statt 'FactorDefinition' einfach anderer name, da es auch für noise gilt !!!!


class Definition:
    def __call__(self):
        return {key: val for (key, val) in self.__dict__.items()}


class GanDefinition(Definition):
    def __init__(self, hidden_units_gen: List[int], hidden_units_disc: List[int], latent_dim: List[int], lr_gen: float,
                 lr_disc: float, output_dim: int):
        self.hidden_units_gen = hidden_units_gen
        self.hidden_units_disc = hidden_units_disc
        self.latent_dim = latent_dim
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.output_dim = output_dim


class Parameter(Definition):
    def __init__(self, lower_bound, upper_bound, discrete, mixture, num_mixture_components, **kwargs):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.discrete = discrete
        self.mixture = mixture
        self.num_mixture_components = num_mixture_components


class ParameterDefinition(Definition):
    def __init__(self, **kwargs):
        for (key, val) in kwargs.items():
            self.__dict__.update({key: Parameter(**val)})


class DesignDefinition(Definition):
    def __init__(self, **kwargs):
        for (key, val) in kwargs.items():
            self.__dict__.update({key: val})


class ResponseDefinition(Definition):
    def __init__(self, **kwargs):
        for (key, val) in kwargs.items():
            self.__dict__.update({key: val})


class Settings:
    def __init__(self, factor_gan_parameter: Dict,
                 noise_gan_parameter: Dict,
                 factor_definition: Dict,
                 noise_definition: Dict,
                 factor_design_definition: Dict,
                 noise_design_definition: Dict,
                 response_definition: Dict):
        # calculate gan output dimension:
        noise_gan_parameter.update({"output_dim": sum([param["num_mixture_components"] \
                                                           if param["mixture"] else 1 for (name, param) in
                                                       noise_definition.items()]) \
                                                  * noise_design_definition["num_samples"]})
        factor_gan_parameter.update({"output_dim": sum([param["num_mixture_components"] \
                                                            if param["mixture"] else 1 for (name, param) in
                                                        factor_definition.items()])})

        self.factor_gan_parameter = GanDefinition(**factor_gan_parameter)
        self.noise_gan_parameter = GanDefinition(**noise_gan_parameter)
        self.factor_definition = ParameterDefinition(**factor_definition)
        self.noise_definition = ParameterDefinition(**noise_definition)
        self.factor_design_definition = DesignDefinition(**factor_design_definition)
        self.noise_design_definition = DesignDefinition(**noise_design_definition)
        self.response_definition = ResponseDefinition(**response_definition)
