from typing import List

class ParameterConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class NoiseConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class StaticParameter:
    SIM_TIME : int = 5*8*60
    JOB_INTERVALL : float = 2.
    MACHINE_PROCESSING_MEAN : List[float] = [1, 3, 5, 7, 9]
    NUM_MACHINES : int = 3
    TESTING_TIME_MEAN : float = 3.
    ERROR_RATE : float = 0.1
    BUFFER_SIZE : int = 1
    
class StaticNoise:
    foo = 'bar'