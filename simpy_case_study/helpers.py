from scipy.stats import expon, norm, uniform
from typing import List, Dict


def sample_expon_dist(size: int = 1, loc: float = 0, scale: float = 1):
    return expon.rvs(loc=loc, scale=scale, size=size).item()


def sample_gaussian_dist(size: int = 1, loc: float = 0, scale: float = 1, lower_limit: float = 0):
    return max(norm.rvs(loc=loc, scale=scale, size=size).item(), lower_limit)


def sample_uniform_dist(size: int = 1, lower: float = 0, upper: float = 1):
    return uniform.rvs(loc=lower, scale=upper, size=size).item()


class SamplingHelpers:
    expon_dist = sample_expon_dist
    gaussian_dist = sample_gaussian_dist
    uniform_dist = sample_uniform_dist


class StaticParameter:
    JOB_INTERVALL: float = 2.
    MACHINE_PROCESSING_MEAN: List[float] = [1, 3, 5, 7, 9, 1]
    TESTING_TIME_MEAN: float = 4.
    ERROR_RATE: float = 0.1
    SIM_TIME: int = 5*8*60
    REPLICATIONS: int = 5


class Targets:
    def __init__(self):
        self.CYCLE_TIME = []
        self.THROUGHPUT = []
