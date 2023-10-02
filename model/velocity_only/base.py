import numpy as np
from scipy.spatial.distance import \
    cdist
from scipy.stats import norm
from scipy.special import softmax

def square_exponential_kernel(x, alpha, length):
    return alpha**2 * np.exp(-0.5 * cdist(x.reshape(-1, 1), x.reshape(-1, 1), 'sqeuclidean')/length**2)


np.random.seed(123)

n_timestep = 10
n_velocity = 20
n_action = 2
n_position = 50
min_position, max_position = 0.0, 4.0
min_velocity, max_velocity = -2.0, 4.0
min_timestep, max_timestep = 0.0, 1.0

timestep = np.linspace(min_timestep, max_timestep, n_timestep)

velocity = np.linspace(min_velocity, max_velocity, n_velocity)
action = np.arange(n_action)
position = np.linspace(min_position, max_position,n_position)

friction_factor = 0.5

n_sample_run = 20

mu = 0.5 + 0.5*np.cos(6*(timestep + 5))
sigma = square_exponential_kernel(timestep, 0.05,  0.1)
own_force = np.random.multivariate_normal(mu, sigma, size=300)

mu = 0.4 + 2 * np.cos(3 * (timestep - 2))
sigma = square_exponential_kernel(timestep, 0.05,  0.1)
push_effect = np.random.multivariate_normal(mu, sigma, size=300)

pref_mu = 2.0
pref_sigma = 0.5

transition_position_sigma = 0.05

# Compute preferences ------------------------------------------------------------------------------------

# p = norm.cdf(
#     position,
#     loc=pref_mu,
#     scale=pref_sigma)
# p /= p.sum()
# log_prior = np.log(p)
log_prior = np.log(softmax(np.arange(n_position)))

# Compute velocity transitions --------------------------------------------------------------------------


def build_transition_velocity_tavv():

    tr = np.zeros((n_timestep, n_action, n_velocity, n_velocity))
    n_sample = push_effect.shape[0]
    bins = list(velocity) + [velocity[-1] + (velocity[-1] - velocity[-2])]

    after_friction = velocity - friction_factor*velocity  # Shape=(n_velocity,)
    after_friction = np.tile(after_friction, (n_action, n_sample, n_timestep, 1))  # Shape=(n_action, n_sample, n_timestep, n_velocity,)

    action_effect = np.tile(push_effect, (n_action, n_velocity, 1, 1, ))  # Shape=(n_action, n_velocity, n_sample, n_timestep,)
    action_effect = np.moveaxis(action_effect, 1, -1)                   # Shape=(n_action, n_sample, n_timestep, n_velocity,)
    action_effect[0] = 0  # Taking action 0 has no effect, taking action 1 is pushing

    own_force_tiled = np.tile(own_force, (n_action, n_velocity, 1, 1, ))        # Shape=(n_action, n_velocity, n_sample, n_timestep,)
    own_force_tiled = np.moveaxis(own_force_tiled, 1, -1)                      # Shape=(n_action, n_sample, n_timestep, n_velocity,)

    new_v = after_friction + action_effect + own_force_tiled
    new_v = np.clip(new_v, bins[0], bins[-1])

    for v_idx, v in enumerate(velocity):
        for a_idx, a in enumerate(action):
            for t_idx, t in enumerate(timestep):
                hist, bins = np.histogram(new_v[a, :, t_idx, v_idx], bins=bins)
                density = hist / np.sum(hist)
                assert np.isclose(np.sum(density), 1.0), "Density does not sum to 1"
                tr[t_idx, a, v_idx, :] = density

    return tr


transition_velocity_tavv = build_transition_velocity_tavv()

# Compute position transitions --------------------------------------------------------------------------


def build_transition_position_pvp():
    tr = np.zeros((n_position, n_velocity, n_position))
    for p_idx, p in enumerate(
            position):
        for v_idx, v in enumerate(
                velocity):
            tr[p_idx, v_idx, :] = norm.pdf(position, loc=p + (1 / n_timestep) * v, scale=transition_position_sigma)
            sum_transition = tr[p_idx, v_idx, :].sum()
            assert sum_transition > 0, "Sum of transition probabilities is 0"
            tr[p_idx, v_idx, :] /= sum_transition
    return tr


transition_position_pvp = build_transition_position_pvp()
