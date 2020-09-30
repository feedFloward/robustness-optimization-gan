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


def scale_to_normal_range(value, param_definition):
    return (((value - param_definition.lower_bound) / (param_definition.upper_bound - param_definition.lower_bound)) * 2) - 1


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


def retransform(feedback, maker_object):
    if type(maker_object).__name__ == "DesignMaker":
        feedback = np.array([feedback])
    elif type(maker_object).__name__ == 'NoiseDesignMaker':
        feedback = np.array(feedback.state)
    
    parameter = maker_object.parameter
    retransformed_feedback = []
    for conf in feedback:
        tmp_list = []
        for (param_name, value) in conf.items():
            value = np.array(value)
            if param_name in parameter.__dict__.keys():
                curr_param_definition = parameter.__dict__[param_name]
                # scale to (-1, 1) range:
                value = scale_to_normal_range(value, curr_param_definition)                
                # dict -> list & unpack mixture lists
                if parameter.__dict__[param_name].mixture:
                    for val in value:
                        tmp_list.append(val)
                else:
                    tmp_list.append(value)

        retransformed_feedback.append(tmp_list)
    return retransformed_feedback


def larger_the_better(response_vals):
    # rounds sn ratio to 3 decimal places
    return round(-10 * np.log(sum(1 / np.array(response_vals)**2)/len(response_vals)), 3)


def smaller_the_better(response_vals):
    return -10 * np.log(sum(np.array(response_vals)**2)/len(response_vals))


def get_sn_calc_func(target, value, **kwargs):
    if target == 'max':
        return larger_the_better
    elif target == 'min':
        return smaller_the_better


def check_config_equal(conf1, conf2):
    # remove response and sn_ration keys
    for conf in [conf1, conf2]:
        conf.pop('response', None)
        conf.pop('sn_ratio', None)

    num_keys = len(conf1.keys())

    return len([conf1_el for conf1_el, conf2_el in zip(conf1.items(), conf2.items()) if conf1_el == conf2_el]) == num_keys


def check_if_config_in_design(new_config, design):
    iterator = iter(design)
    try:
        return any(check_config_equal(new_config, rest) for rest in iterator)
    except StopIteration:
        return False


def reshape_to_design(values_array, num_configs):
    return np.reshape(values_array, newshape= (num_configs, -1))


def flatten(values):
    return np.reshape(values, newshape= (1, -1))
