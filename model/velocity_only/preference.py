from scipy.stats import norm

from . base import *


def build():
    target = 1.0
    tolerance = 0.5

    # Compute log prior
    p = norm.cdf(
        position,
        loc=target,
        scale=tolerance)
    p /= p.sum()
    log_prior = np.log(
        p)
    return log_prior
