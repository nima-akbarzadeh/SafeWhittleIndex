from Markov import *
from whittle import *

# Basic Parameters
n_steps = 100
n_states = 3
u_type = 2
u_order = 8
n_arms = 5
thresholds = 0.5 * np.ones(n_arms)
transition_type = 3
function_type = np.ones(n_arms, dtype=np.int32)
n_episodes = 100
initial_states = (n_states - 1) * np.ones(n_arms, dtype=np.int32)

if transition_type == 0:
    prob_remain = np.round(np.linspace(0.1, 0.9, n_arms), 2)
    np.random.shuffle(prob_remain)
elif transition_type == 1:
    prob_remain = np.round(np.linspace(0.05, 0.45, n_arms), 2)
    np.random.shuffle(prob_remain)
elif transition_type == 2:
    prob_remain = np.round(np.linspace(0.05, 0.45, n_arms), 2)
    np.random.shuffle(prob_remain)
elif transition_type == 3:
    prob_remain = np.round(np.linspace(0.1 / n_states, 1 / n_states, n_arms), 2)
    np.random.shuffle(prob_remain)
else:
    prob_remain = np.round(np.linspace(0.1, 0.9, n_arms), 2)
    np.random.shuffle(prob_remain)

reward_increasing = True
transition_increasing = True
n_choices = 1

# Basic Parameters
R = Values(n_steps, n_arms, n_states, function_type, reward_increasing)
M = MarkovDynamics(n_arms, n_states, prob_remain, transition_type, transition_increasing)
reward_bandits = R.vals
transition_bandits = M.transitions
n_trials_neutrl = n_arms * n_states * n_steps
n_trials_safety = n_arms * n_states * n_steps
method = 3
max_wi = 1

if __name__ == '__main__':

    # SWA = SafeWhittle(n_states, n_arms, reward_bandits, transition_bandits, 5, u_type, u_order, thresholds)

    n_parts = 100
    time_steps = 100
    SWL = SafeWhittleLight([n_states, n_parts], n_arms, reward_bandits, transition_bandits, time_steps, u_type, u_order,
                           thresholds)
    lambda_valus = [0.02]
    for lmbda in lambda_valus:
        # pi0, _, _ = SWA.backward_discreteliftedstate(0, lmbda)
        pi, _, _ = SWL.backward_discreteliftedstate(0, lmbda)
        print('-'*20 + ' '*5 + str(lmbda))
        print('=' * 5 + ' t: ' + str(0))
        print(pi[:, :, 0])
        # for t in range(time_steps):
        #     print('='*5 + ' t: ' + str(t))
        #     print(pi0[:, :, t])
        #     print(pi[:, :, t])
        # print('='*5 + ' t: ' + str(1))
        # print(pi[:, :, 1])
        # print('='*5 + ' t: ' + str(time_steps-1))
        # print(pi[:, :, time_steps-1])

