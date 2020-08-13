import importlib
from robustness_optimization.interface.api import ModelInterface

class SimpyModel(ModelInterface):
    def __init__(self, model_path : str):
        '''
        pkg_path : absolute path to simulations model's package repository
        '''
        spec = importlib.util.spec_from_file_location("", model_path)
        self.model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.model)