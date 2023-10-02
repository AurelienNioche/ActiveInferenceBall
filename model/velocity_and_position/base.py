import numpy as np
from scipy.special import softmax
from scipy.stats import norm
from model.helpers import square_exponential_kernel, normalize_last_column

np.random.seed(123)

n_timestep = 6
n_velocity = 20
n_action = 2
n_position = 50

min_velocity, max_velocity = 0., 3.0
min_position, max_position = 0.0, 2.0
min_timestep, max_timestep = 0.0, 1.0

timestep = np.linspace(min_timestep, max_timestep, n_timestep)
position = np.linspace(min_position, max_position, n_position)
velocity = np.linspace(min_velocity, max_velocity, n_velocity)
action = np.arange(n_action)

friction_factor = 0.5

n_sample = 400


# Compute force that depends both on context (timestep) and position ------------------------------

x = np.column_stack([x.ravel() for x in np.meshgrid(timestep, position)])

mu = np.zeros((2, n_timestep * n_position))
mu[0] = 0.8 + 0.6 * np.cos(6 * x[:, 0] - 2)  # + 0.3*np.sin(4*x[:,1] + 3)
mu[1] = 0.8 + 0.7 * np.cos(3 * x[:, 0] - 4)  # + 0.3*np.sin(4*x[:,1] + 3)

sigma = square_exponential_kernel(x, alpha=0.05, length=0.1)

force_satp = np.zeros((n_sample, n_action, n_timestep, n_position))
for a in action:
    samples = np.random.multivariate_normal(mu[a], sigma, size=n_sample)
    force_satp[:, a, :, :] = samples.reshape(n_sample, n_timestep, n_position)


# compute preferences ------------------------------------------------------------------------------------

log_prior = np.log(softmax(np.arange(n_position)))


# Compute velocity transitions --------------------------------------------------------------------------


def build_transition_velocity_atpvv():

    tr = np.zeros((n_action, n_timestep, n_position, n_velocity, n_velocity))
    bins = list(velocity) + [velocity[-1] + (velocity[-1] - velocity[-2])]
    after_friction = velocity - friction_factor * velocity  # Shape=(n_velocity,)
    exp_after_friction = np.expand_dims(after_friction[:],
                                        tuple(range(len(force_satp.shape))))  # Shape=(1, 1, 1, 1, n_velocity)
    exp_force = np.expand_dims(force_satp, -1)   # Shape=(n_sample, n_action, n_timestep, n_position, 1)
    new_v = exp_after_friction + exp_force  # Shape=(n_sample, n_action, n_timestep, n_position, n_velocity)
    new_v = np.clip(new_v, min_velocity, max_velocity)
    for a_idx in range(n_action):
        for t_idx in range(n_timestep):
            for p_idx in range(n_position):
                for v_idx in range(n_velocity):
                    tr[a_idx, t_idx, p_idx, v_idx, :], _ = np.histogram(new_v[:, a_idx, t_idx, p_idx, v_idx], bins=bins)
    return normalize_last_column(tr)


transition_velocity_atpvv = build_transition_velocity_atpvv()


# Compute position transitions --------------------------------------------------------------------------


def build_transition_position_pvp():

    tr = np.zeros((n_position, n_velocity, n_position))
    dt = (max(timestep) - min(timestep)) / (n_timestep - 1)
    dp = (max(position) - min(position)) / (n_position - 1)
    for p_idx, p in enumerate(position):
        for v_idx, v in enumerate(velocity):
            tr[p_idx, v_idx, :] = norm.pdf(position, loc=p + dt * v, scale=dp / 4)
    return normalize_last_column(tr)


transition_position_pvp = build_transition_position_pvp()
