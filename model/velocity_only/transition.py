from scipy.spatial.distance import cdist
from scipy.stats import norm

from . base import *


def square_exponential_kernel(x, alpha, length):
    return alpha**2 * np.exp(-0.5 * cdist(x.reshape(-1, 1), x.reshape(-1, 1), 'sqeuclidean')/length**2)


def sample_action_effect(t, size=1):
    mu = 0.4 + 1.2 * np.cos(3 * (t - 2))
    alpha = 0.05
    length = 0.1
    sigma = square_exponential_kernel(t, alpha, length)
    return np.random.multivariate_normal(mu, sigma, size=size)


def sample_own_force(t, size=1):

    mu = 0.5 + 0.5*np.cos(6*(t + 5))
    alpha = 0.05
    length = 0.1
    sigma = square_exponential_kernel(t, alpha, length)
    return np.random.multivariate_normal(mu, sigma, size=size)


def build():

    n_sample = 300
    # Building the velocity transaction matrix
    own_force = sample_own_force(timestep, size=n_sample)
    action_effect = sample_action_effect(timestep, size=n_sample)

    # Compute the 'true' transition probabilities for the velocity
    transition_velocity_tavv = np.zeros((n_timestep, n_action, n_velocity, n_velocity))
    for v_idx, v in enumerate(velocity):
        for t_idx, t in enumerate(timestep):
            for a in action:
                new_v = np.zeros(n_sample)
                new_v += v - friction_factor*v
                new_v += action_effect[:, t_idx]*a
                new_v += own_force[:, t_idx]
                new_v = np.clip(new_v, -min(velocity), max(velocity))
                hist, bins = np.histogram(
                    new_v,
                    bins=list(velocity) + [velocity[-1] + (velocity[-1] - velocity[-2])])
                density = hist / np.sum(hist)

                transition_velocity_tavv[t_idx, a, v_idx, :] = density

    transition_position_pvp = np.zeros((n_position, n_velocity, n_position))
    for p_idx, p in enumerate(
            position):
        for v_idx, v in enumerate(
                velocity):
            for p2_idx, p2 in enumerate(
                    position):
                loc = p + (1 / n_timestep) * v
                print(loc)
                transition_position_pvp[
                    p_idx, v_idx, p2_idx
                ] = norm.pdf(
                    p2,
                    loc=loc,
                    scale=0.01
                )
            sum_transition = transition_position_pvp[p_idx, v_idx, :].sum()
            assert sum_transition > 0, "Sum of transition probabilities is 0"
            transition_position_pvp[p_idx, v_idx, :] /= sum_transition

    return transition_velocity_tavv, transition_position_pvp
