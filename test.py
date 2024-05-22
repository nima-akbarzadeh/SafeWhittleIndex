from whittle import *
from processes import *
from learning import *
import matplotlib.pyplot as plt
import joblib


if __name__ == '__main__':

    # Basic Parameters
    n_steps = 5
    n_states = 3
    n_arms = 2
    n_coeff = 1
    u_type = 1
    u_order = 16
    thresholds = 0.5 * np.ones(n_arms)
    choice_fraction = 0.5

    transition_type = 3
    function_type = np.ones(n_arms, dtype=np.int32)

    n_episodes = 100

    reward_increasing = True
    transition_increasing = True
    max_wi = 1

    # Simulation Parameters
    n_choices = np.maximum(1, int(choice_fraction * n_arms))
    initial_states = (n_states - 1) * np.ones(n_arms, dtype=np.int32)

    # Basic Parameters
    R = Values(n_steps, n_arms, n_states, function_type, reward_increasing)
    reward_bandits = R.vals

    n_trials_neutrl = n_arms * n_states * n_steps
    n_trials_safety = n_arms * n_states * n_steps
    method = 3

    res_dict = {}
    for i in range(100):
        print(i)
        prob_remain = np.array([np.round(random.uniform(0.1 / n_states, 1 / n_states), 2) for _ in range(n_arms)])
        M = MarkovDynamics(n_arms, n_states, prob_remain, transition_type, transition_increasing)
        transition_bandits = M.transitions
        SafeW = SafeWhittle(n_states, n_arms, reward_bandits, transition_bandits, n_steps, u_type, u_order, thresholds)
        SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        sw_bandits = SafeW.w_indices
        _, obj_s, _ = Process_SafeRB(SafeW, sw_bandits, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
        res_dict[tuple(prob_remain)] = np.mean(obj_s)

    max_key = max(res_dict, key=lambda k: res_dict[k])
    max_value = res_dict[max_key]
    print(f"Max: Key = {max_key}, Value = {max_value}")
    min_key = min(res_dict, key=lambda k: res_dict[k])
    min_value = res_dict[min_key]
    print(f"Min: Key = {min_key}, Value = {min_value}")
