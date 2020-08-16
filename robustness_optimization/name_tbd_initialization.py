'''
- reads config file
- sets all optimization parameter, gan parameter, simulation parameter etc in extra CLASS

!!! name des files und ort (welcher unterordner sind zu bestimmen)
'''
import json
from robustness_optimization.types.optimization_types import Settings

SETTINGS_PATH = "settings.json" # muss immer im root ordner 'robustness-optimization-gan' liegen

def read_settings():
    with open(SETTINGS_PATH, 'r') as f:
        settings = json.load(f)
    return Settings(**settings)