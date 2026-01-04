import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def action_value(mean=0.0, std=1.0):
    return np.random.normal(mean, std)

def reward(q, std=1.0):
    return np.random.normal(q, std)

def run_simulation(k, eps):
    num_steps = 1000
    reward_per_steps = np.array([0.0] * num_steps)

    q_true = [0.0] * k
    for i in range(k):
        q_true[i] = action_value() # gauss 
    Qt = [0.0] * k
    sum_r_action = [0] * k
    cnt_action = [0] * k
    num_opt_action = np.array([0] * num_steps)
    percent_opt_action = np.array([0.0] * num_steps)

    for step_idx in range(1, num_steps+1):
        # Take action
        action = None
        if np.random.rand() < eps or step_idx == 1:
            action = np.random.randint(k) #[1, k+1)
        else:
            best_q = np.max(Qt)
            action = np.random.choice([i for i, q in enumerate(Qt) if q == best_q])

        r = reward(q_true[action]) # get reward mean = q*(a), variance = 1
        # Calculate Qt
        cnt_action[action] += 1
        sum_r_action[action] += r
        reward_per_steps[step_idx-1] = r
        Qt[action] = sum_r_action[action] / cnt_action[action] 
        if step_idx == 1:
            num_opt_action[step_idx-1] = action == np.argmax(q_true)
        else:
            num_opt_action[step_idx-1] = num_opt_action[step_idx-2] + (action == np.argmax(q_true)) # for fig 2.3
        percent_opt_action[step_idx-1] = num_opt_action[step_idx-1] / step_idx # for fig 2.3

    return reward_per_steps, percent_opt_action

if __name__ == "__main__":
    k = 10
    num_runs = 2000
    eps_list = [0.0, .01, .1]
    colors = ["green", "red", "blue"]
    percent_opt_action_list = np.array([[0.0]*1000 for _ in range(len(eps_list))])
    avg_reward_list = np.array([[0.0]*1000 for _ in range(len(eps_list))])

    plt.figure(figsize=(12, 8))
    for eps_idx, eps in enumerate(eps_list):
        print(f"Running simulation for eps={eps}")
        for run_idx in tqdm(range(1, num_runs+1)):
            reward_per_steps, percent_opt_action = run_simulation(k, eps)
            avg_reward_list[eps_idx] += (reward_per_steps - avg_reward_list[eps_idx]) / run_idx # avg over runs
            percent_opt_action_list[eps_idx] += (percent_opt_action - percent_opt_action_list[eps_idx]) / run_idx # avg over runs
        plt.plot(avg_reward_list[eps_idx], color=colors[eps_idx], label=f"eps={eps}")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.hlines(y=0, xmin=0, xmax=1000, color='k', linestyle='--')
    plt.xlim(0, 1000)
    plt.ylim(0, 1.5)
    plt.title("Average Reward vs Steps for Different Epsilon Values")
    plt.legend()
    plt.savefig("./outputs/fig22_1.jpg")
    # plt.show()

    plt.figure(figsize=(12, 8))
    for eps_idx in range(len(eps_list)):
        plt.plot(percent_opt_action_list[eps_idx], color=colors[eps_idx], label=f"eps={eps_list[eps_idx]}")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.xlim(0, 1000)
    plt.ylim(0, 1)
    plt.title("Percentage of Optimal Action vs Steps for Different Epsilon Values")
    plt.legend()
    plt.savefig("./outputs/fig22_2.jpg")
    # plt.show()