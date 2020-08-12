import sys
import argparse
from case_study.sim_types import ParameterConfig, StaticParameter, NoiseConfig, StaticNoise
from case_study.model import simulation_run

'''
DAS GANZE FILE KOMMT INS OPTIMIZATION PACKAGE UNTER EINEN SUBFOLDER 'API' !!!!!!!!
'''

'''
This file provides wrappers for functions of the simpy model.
'''

def set_factors(**kwargs):
    return ParameterConfig(**kwargs)

def set_noise(**kwargs):
    return NoiseConfig(**kwargs)

def run(factor_config, noise_config):
    '''
    Performs one single simulation run with parameters and noise set in...
    '''
    simulation_run(factor_config= set_factors(**factor_config), noise_config= set_noise(**noise_config))


def main(factor_design, noise_design):
    '''
    performs one complete iteration of GAN optimization
    '''
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-factor_dict", help="path to json with factor name and value as key value pairs", type=str)
    parser.add_argument("-noise_dict", type=str)
    parser.add_argument("-simulation_settings", type=str)
    args = parser.parse_args()
    print(args.factor_dict)
    print(args.noise_dict)
    print(args.simulation_settings)
    '''
    json files auslesen und main mit argumenten aufrufen...
    '''
    print('...ended')
    main(args)