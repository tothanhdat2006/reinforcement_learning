import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10, eps=0.0, initial_q=0.0, sample_average=True, step_size=0.1, random_walk=False):
        self.k = k

        self.q_true = [np.random.normal(0.0, 1.0) + initial_q for _ in range(k)]
        self.Qt = [0.0] * k
        self.cnt_action = [0] * k

        self.sample_average = sample_average
        self.step_idx = 0
        self.step_size = step_size

        self.eps = eps
        self.initial_q = initial_q
        self.random_walk = random_walk

    def get_reward(self, action):
        if self.random_walk:
            return np.random.normal(self.q_true[action], 1.0) + np.random.normal(0.0, 0.01)
        else:
            return np.random.normal(self.q_true[action], 1.0)

    def reset(self):
        self.q_true = [np.random.normal(0.0, 1.0) + self.initial_q for _ in range(self.k)]
        self.Qt = [0.0] * self.k
        self.cnt_action = [0] * self.k

    def act(self):
        if np.random.rand() < self.eps or self.step_idx == 1:
            return np.random.randint(self.k)
        else:
            best_q = np.max(self.Qt)
            return np.random.choice([i for i, q in enumerate(self.Qt) if q == best_q])

    def step(self, action):
        reward = self.get_reward(action)
        self.cnt_action[action] += 1
        if self.sample_average:
            self.Qt[action] = self.Qt[action] + (reward - self.Qt[action]) / self.cnt_action[action] # sample average: new = old + 1/n * (R - old)
        else:
            self.Qt[action] += self.step_size * (reward - self.Qt[action]) # new = old + step_size * (R - old)
        self.step_idx += 1
        return reward


def run_simulation(bandit, num_steps=1000):
    bandit.reset()
    num_steps = num_steps
    reward_per_steps = np.array([0.0] * num_steps)
    num_opt_action = np.array([0] * num_steps)
    percent_opt_action = np.array([0.0] * num_steps)

    for step_idx in range(1, num_steps + 1):
        action = bandit.act()
        bandit.step(action)

        reward_per_steps[step_idx - 1] = bandit.Qt[action] 
        if step_idx == 1:
            num_opt_action[step_idx - 1] = action == np.argmax(bandit.q_true)
        else:
            num_opt_action[step_idx - 1] = num_opt_action[step_idx - 2] + (action == np.argmax(bandit.q_true))
        percent_opt_action[step_idx - 1] = num_opt_action[step_idx - 1] / step_idx 

    return reward_per_steps, percent_opt_action

def figure22():
    # sample average, incrementally computed
    k = 10
    num_runs = 2000
    eps_list = [0.0, .01, .1]
    colors = ["green", "red", "blue"]
    percent_opt_action_list = np.array([[0.0] * 1000 for _ in range(len(eps_list))])
    avg_reward_list = np.array([[0.0] * 1000 for _ in range(len(eps_list))])
    bandit = [Bandit(k=k, eps=eps, initial_q=0.0, sample_average=True, step_size=None) for eps in eps_list]

    plt.figure(figsize=(12, 8))
    for eps_idx, eps in enumerate(eps_list):
        for run_idx in tqdm(range(1, num_runs + 1), desc=f"eps={eps}"):
            reward_per_steps, percent_opt_action = run_simulation(bandit[eps_idx])
            avg_reward_list[eps_idx] += (reward_per_steps - avg_reward_list[eps_idx]) / run_idx # avg over runs
            percent_opt_action_list[eps_idx] += (percent_opt_action - percent_opt_action_list[eps_idx]) / run_idx # avg over runs

        plt.plot(avg_reward_list[eps_idx], color=colors[eps_idx], label=f"eps={eps} Average Reward")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.hlines(y=0, xmin=0, xmax=1000, color='k', linestyle='--')
    plt.xlim(0, 1000)
    plt.ylim(0, 1.5)
    plt.title("Average Reward vs Steps for Different Epsilon Values")
    plt.legend()
    plt.grid()
    plt.savefig("./outputs/fig22_1.jpg")
    

    plt.figure(figsize=(12, 8))
    for eps_idx, eps in enumerate(eps_list):
        plt.plot(percent_opt_action_list[eps_idx], color=colors[eps_idx], label=f"eps={eps} % Optimal Action")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.xlim(0, 1000)
    plt.ylim(0, 1)
    plt.title("Percentage of Optimal Action vs Steps for Different Epsilon Values")
    plt.legend()
    plt.grid()
    plt.savefig("./outputs/fig22_2.jpg")


def figure23():
    # sample average + constant step size
    # eps = 0.1, 1 run = 10000 steps
    k = 10
    num_runs = 2000
    num_steps = 10000
    eps = 0.1
    step_size = 0.1
    bandit_sample_avg = Bandit(k=k, eps=eps, initial_q=0.0, sample_average=True, step_size=None, random_walk=True)
    bandit_const_step = Bandit(k=k, eps=eps, initial_q=0.0, sample_average=False, step_size=step_size, random_walk=True)

    # sample average
    plt.figure(figsize=(12, 8))
    print("Running simulation for Sample Average")
    avg_reward_list = np.array([0.0] * num_steps)
    avg_percent_opt_action_list = np.array([0.0] * num_steps)
    for run_idx in tqdm(range(1, num_runs + 1), desc=f"eps={eps}"):
        reward_per_steps_avg, percent_opt_action_avg = run_simulation(bandit_sample_avg, num_steps=num_steps)
        avg_reward_list += (reward_per_steps_avg - avg_reward_list) / run_idx # avg over runs
        avg_percent_opt_action_list += (percent_opt_action_avg - avg_percent_opt_action_list) / run_idx # avg over runs
    plt.plot(avg_reward_list, color="blue", label="Sample Average")

    # constant step size
    const_reward_list = np.array([0.0] * num_steps)
    const_percent_opt_action_list =  np.array([0.0] * num_steps)
    print("Running simulation for Constant Step Size")
    for run_idx in tqdm(range(1, num_runs + 1), desc=f"eps={eps}"):
        reward_per_steps_const, percent_opt_action_const = run_simulation(bandit_const_step, num_steps=num_steps)
        const_reward_list += (reward_per_steps_const - const_reward_list) / run_idx # avg over runs
        const_percent_opt_action_list += (percent_opt_action_const - const_percent_opt_action_list) / run_idx # avg over runs
    plt.plot(const_reward_list, color="red", label="Constant Step Size (0.1)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.hlines(y=0, xmin=0, xmax=num_steps, color='k', linestyle='--')
    plt.xlim(0, num_steps)
    plt.ylim(0, 1.5)
    plt.title("Average Reward vs Steps: Sample Average vs Constant Step Size")
    plt.legend()
    plt.savefig("./outputs/ex_2_5_fig_avg_reward.jpg")

    plt.figure(figsize=(12, 8))
    plt.plot(avg_percent_opt_action_list, color="purple", label="% Optimal Action")
    plt.plot(const_percent_opt_action_list, color="orange", label="% Optimal Action (Constant Step Size)")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.xlim(0, num_steps)
    plt.ylim(0, 1)
    plt.title("Percentage of Optimal Action vs Steps: Sample Average vs Constant Step Size")
    plt.legend()
    plt.savefig("./outputs/ex_2_5_fig_percent_opt.jpg")


if __name__ == "__main__":
    # figure22()
    figure23()