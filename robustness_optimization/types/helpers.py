import numpy as np

def uniform(length):
    return np.random.uniform(-1, 1, length)

def split_up_sampling(sample, param_definition):
    sample_slice_lengths = [param_vals.num_mixture_components if param_vals.mixture else 1 for (param_name, param_vals) in param_definition]
    sample_slices = [sum(sample_slice_lengths[:index+1]) for (index, slice_len) in enumerate(sample_slice_lengths)]
    sample_slices.insert(0, 0)
    sample_sliced = [sample[sample_slices[i]: sample_slices[i+1]] for i in range(len(sample_slices)-1)]
    return sample_sliced

def scale_to_param_range(value, param_definition):
    return param_definition.lower_bound + ((value + 1) / 2) * (param_definition.upper_bound - param_definition.lower_bound)

def make_discrete(value):
    return np.rint(value)

def normalize(value):
    return value / sum(value)

def type_casting(value, param_definition):
    if param_definition.mixture:
        return value.tolist()
    elif param_definition.discrete:
        return int(value)
    else:
        return float(value)