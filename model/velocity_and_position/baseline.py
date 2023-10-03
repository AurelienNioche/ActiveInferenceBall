from model.velocity_and_position.base import *


def run(n_sample=20):

    policies = ["all-zero", "all-one", "random", "max-expected-velocity"]
    results = []

    for policy in policies:
        hist_pos = np.zeros((n_sample, n_timestep))
        hist_vel = np.zeros_like(hist_pos)

        for sample in range(n_sample):
            p_idx = np.absolute(position).argmin()  # Something close to 0
            v_idx = np.absolute(velocity).argmin()  # Something close to 0

            np.random.seed(123 + sample * 123)

            for t_idx, t in enumerate(timestep):
                if policy == "all-one":
                    a = 1
                elif policy == "all-zero":
                    a = 0
                elif policy == "random":
                    a = np.random.choice([0, 1])
                elif policy == "max-expected-velocity":
                    e_v = np.zeros(2)
                    for a in range(2):
                        e_v[a] = np.average(
                            velocity,
                            weights=transition_velocity_atpvv[a, t_idx, p_idx, v_idx, :])
                    a = e_v.argmax()
                else:
                    raise ValueError
                tr_vel = transition_velocity_atpvv[a, t_idx, p_idx, v_idx, :]
                v_idx = np.random.choice(np.arange(n_velocity), p=tr_vel)
                tr_pos = transition_position_pvp[p_idx, v_idx, :]
                p_idx = np.random.choice(np.arange(n_position), p=tr_pos)

                hist_pos[sample, t_idx] = position[p_idx]
                hist_vel[sample, t_idx] = velocity[v_idx]

        results.append({
            "policy": policy,
            "position": hist_pos,
            "velocity": hist_vel})
    return results

