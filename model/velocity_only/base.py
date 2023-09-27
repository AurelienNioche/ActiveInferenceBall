import numpy as np

n_timestep = 6
n_velocity = 40
n_action = 2
n_position = 50
min_position, max_position = -100.0, 100.0
min_velocity, max_velocity = -100.0, 100.0

timestep = np.linspace(
    0,
    1.0,
    n_timestep)
velocity = np.linspace(
    min_velocity,
    max_velocity,
    n_velocity)
action = np.arange(
    n_action)
position = np.linspace(
    min_position,
    max_position,
    n_position)

# max_velocity = 10.0
friction_factor = 0.5

n_sample_run = 20
