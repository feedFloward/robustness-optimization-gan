from scipy.stats import expon, norm, uniform

def sample_expon_dist(size: int = 1, loc : float = 0, scale : float = 1):
    return expon.rvs(loc=loc, scale=scale, size=size).item()

def sample_gaussian_dist(size: int = 1, loc : float = 0, scale : float = 1, lower_limit : float = 0):
    return max(norm.rvs(loc=loc, scale=scale, size=size).item(), lower_limit)

def sample_uniform_dist(size: int = 1, lower: float = 0, upper: float = 1):
    return uniform.rvs(loc= lower, scale= upper, size= size).item()

class SamplingHelpers:
    expon_dist = sample_expon_dist
    gaussian_dist = sample_gaussian_dist
    uniform_dist = sample_uniform_dist