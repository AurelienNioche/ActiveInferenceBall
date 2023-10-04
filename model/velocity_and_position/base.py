import numpy as np
from scipy.special import softmax
from scipy.stats import norm
from model.helpers import square_exponential_kernel, normalize_last_column

np.random.seed(123)

n_timestep = 10
n_velocity = 10
n_action = 2
n_position = 10

min_velocity, max_velocity = -1., 2.
min_position, max_position = 0.0, 1.1
min_timestep, max_timestep = 0.0, 1.0

timestep = np.linspace(min_timestep, max_timestep, n_timestep)
position = np.linspace(min_position, max_position, n_position)
velocity = np.linspace(min_velocity, max_velocity, n_velocity)
action = np.arange(n_action)

friction_factor = 0.8

dp = (max_position - min_position) / (n_position - 1)
sigma_transition_position = 0.01*dp

n_sample = 400

# Compute force that depends both on context (timestep) and position ------------------------------

xt, xp = np.meshgrid(timestep, position, indexing='ij')
xp__flat = xp.ravel()
xt__flat = xt.ravel()

mu = np.zeros((2, n_timestep * n_position))
mu[0] = -0.2 + 0.6 * np.cos(0.4 * xt__flat)  # + 0.3*np.sin(4*x[:,1] + 3)
mu[1] = 0.5 + 1.2 * np.cos(4 * (xt__flat - 0.2))  # + 0.3*np.sin(4*x[:,1] + 3)

sigma = square_exponential_kernel(np.column_stack((xt__flat, xp__flat)),
                                  alpha=0.1, length=0.01)

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
                                        tuple(range(force_satp.ndim)))  # Shape=(1, 1, 1, 1, n_velocity)
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
    dt = (max_timestep - min_timestep) / (n_timestep - 1)
    for p_idx, p in enumerate(position):
        for v_idx, v in enumerate(velocity):
            _mu = p + dt * v
            pdf = norm.pdf(
                position,
                loc=_mu,
                scale=sigma_transition_position)
            sum_pdf = pdf.sum()
            if sum_pdf > 0:
                tr[p_idx, v_idx, :] = pdf / sum_pdf
            else:
                tr[p_idx, v_idx, np.argmin(np.abs(position - _mu))] = 1.
    return tr  # normalize_last_column(tr)


transition_position_pvp = build_transition_position_pvp()
# for p_idx, p in enumerate(position):
#     for v_idx, v in enumerate(velocity):
#         assert np.isclose(np.sum(transition_position_pvp[p_idx, v_idx, :]), 1.0), \
#             "transition_position_pvp is not normalized"

# ------------------- previous settings -------------------
#
# np.random.seed(123)
#
# n_timestep = 6
# n_velocity = 20
# n_action = 2
# n_position = 50
#
# min_velocity, max_velocity = 0., 3.0
# min_position, max_position = 0.0, 2.0
# min_timestep, max_timestep = 0.0, 1.0
#
# timestep = np.linspace(min_timestep, max_timestep, n_timestep)
# position = np.linspace(min_position, max_position, n_position)
# velocity = np.linspace(min_velocity, max_velocity, n_velocity)
# action = np.arange(n_action)
#
# friction_factor = 0.5
#
# dp = (max_position - min_position) / (n_position - 1)
# sigma_transition_position = 0.25*dp
#
# n_sample = 400
#
# # Compute force that depends both on context (timestep) and position ------------------------------
#
# x = np.column_stack([x.ravel() for x in np.meshgrid(timestep, position)])
#
# mu = np.zeros((2, n_timestep * n_position))
# mu[0] = 0.8 + 0.6 * np.cos(6 * x[:, 0] - 2)  # + 0.3*np.sin(4*x[:,1] + 3)
# mu[1] = 0.8 + 0.7 * np.cos(3 * x[:, 0] - 4)  # + 0.3*np.sin(4*x[:,1] + 3)
#
# sigma = square_exponential_kernel(x, alpha=0.05, length=0.1)
