import importlib
import sys

class SimpyModel:
    def __init__(self, model_path : str):
        '''
        pkg_path : absolute path to simulations model's package repository
        '''
        spec = importlib.util.spec_from_file_location("", model_path)
        self.model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.model)
        print(self.model.test)

    def run(self, **kwargs):
        #! Modell muss eine function run haben !
        self.model.run(**kwargs)

    def name_tbd(self, **kwargs):
        self.model.main(**kwargs)