from typing import Dict, List

class Parameter:
    def __call__(self):
        return {key: val for (key, val) in self.__dict__.items()}

class GanParameter(Parameter):
    def __init__(self, hidden_units_gen : List[int], hidden_units_disc : List[int], latent_dim : List[int], lr_gen : float, lr_disc : float, output_dim : int):
        self.hidden_units_gen = hidden_units_gen
        self.hidden_units_disc = hidden_units_disc
        self.latent_dim = latent_dim
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.output_dim = output_dim

class Settings:
    def __init__(self, factor_gan_parameter: Dict, **kwargs):
        self.factor_gan_parameter = GanParameter(**factor_gan_parameter)