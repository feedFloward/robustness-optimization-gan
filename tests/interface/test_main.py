from robustness_optimization.interface import SimpyModel

model_path = "C:/Users/fconrad/git/robustness-optimization-gan/simpy_case_study/model.py"

gan_output_factors = [
    {'num_machines': 1, 'buffer_size': 1},
    {'num_machines': 1, 'buffer_size': 2},
    {'num_machines': 2, 'buffer_size': 1},
    {'num_machines': 2, 'buffer_size': 2},
]
gan_output_noise = [
    [
        {'product_mix': [0.8, 0.1, 0.1]},
        {'product_mix': [0.8, 0.1, 0.1]},
        {'product_mix': [0.1, 0.8, 0.1]},
        {'product_mix': [0.1, 0.1, 0.8]},
        {'product_mix': [0.2, 0.6, 0.2]},
    ],
    [
        {'product_mix': [0.9, 0.05, 0.05]},
        {'product_mix': [0.6, 0.2, 0.2]},
        {'product_mix': [0.05, 0.9, 0.05]},
        {'product_mix': [0.05, 0.05, 0.9]},
        {'product_mix': [0.2, 0.2, 0.6]},
    ]
]
champion_factors = [{'num_machines': 1, 'buffer_size': 1}]
champion_noise = [
    {'product_mix': [0.8, 0.1, 0.1]},
    {'product_mix': [0.8, 0.1, 0.1]},
    {'product_mix': [0.1, 0.8, 0.1]},
    {'product_mix': [0.1, 0.1, 0.8]},
    {'product_mix': [0.2, 0.6, 0.2]},
]

def get_model():
    return SimpyModel(model_path)

def test_main():
    model = get_model()
    model.main(factor_design= gan_output_factors, noise_design= champion_noise, competitor_flag= 'factor')
    model.main(factor_design= champion_factors, noise_design= gan_output_noise, competitor_flag= 'noise')