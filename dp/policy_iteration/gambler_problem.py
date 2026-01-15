import numpy as np
import matplotlib.pyplot as plt

DISCOUNT = 0.9
GOAL = 100
STATES = np.arange(GOAL + 1)

def is_terminal(state: tuple) -> bool:
    return state == 0 or state == GOAL

def value_iteration(PROB_HEAD=0.4):
    theta = 1e-5
    V = np.zeros(GOAL + 1)
    V[0] = 0.0
    V[GOAL] = 1.0
    sweeps_history = []
    while True:
        old_V = V.copy()
        sweeps_history.append(old_V)
        # print(f"Iteration {len(sweeps_history)}")

        for s in STATES[1:GOAL]:
            action_returns = []
            actions = np.arange(min(s, GOAL-s)+1)
            for action in actions:
                action_returns.append(PROB_HEAD * V[s + action] + (1 - PROB_HEAD) * V[s - action])
            V[s] = np.max(action_returns) # max here ~ eval + improve
            
        delta = abs(V - old_V).max()
        if delta < theta:
            sweeps_history.append(V)
            return V, sweeps_history
        
def get_optimal_policy(V, PROB_HEAD=0.4):
    policy = np.zeros(GOAL+1)
    for s in STATES[1:GOAL]:
        action_returns = []
        actions = np.arange(min(s, GOAL-s)+1)
        for action in actions:
            action_returns.append(PROB_HEAD * V[s + action] + (1 - PROB_HEAD) * V[s - action])
        policy[s] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1] # argmax
    return policy

def fig43():
    V, sweeps_history = value_iteration()
    policy = get_optimal_policy(V)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))

    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('./outputs/fig43.png')
    plt.close()

def fig43_1():
    V, sweeps_history = value_iteration(PROB_HEAD=0.25)
    policy = get_optimal_policy(V, PROB_HEAD=0.25)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))

    plt.xlabel('Capital')
    plt.ylabel('Value estimates (p_head = 0.25)')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('./outputs/fig43_1.png')
    plt.close()

def fig43_2():
    V, sweeps_history = value_iteration(PROB_HEAD=0.55)
    policy = get_optimal_policy(V, PROB_HEAD=0.55)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1) 
    for sweep, state_value in enumerate(sweeps_history):
        if sweep > 10 and sweep < len(sweeps_history) - 5: # ~700 sweep
            continue
        plt.plot(state_value, label='sweep {}'.format(sweep))

    plt.xlabel('Capital')
    plt.ylabel('Value estimates (p_head = 0.55)')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('./outputs/fig43_2.png')
    plt.close()



if __name__ == "__main__":
    # fig43()
    fig43_1()
    fig43_2()