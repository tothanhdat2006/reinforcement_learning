import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10, eps=0.0, initial_q=0.0, true_reward=0.0,
                 sample_average=True, UCB=None, gradient=False, gradient_baseline=False, 
                 step_size=0.1, random_walk=False):
        self.k = k

        self.initial_q = initial_q
        self.true_reward = true_reward

        self.q_true = [np.random.normal(true_reward, 1.0) + initial_q for _ in range(k)]
        self.Qt = [self.initial_q] * k 
        self.cnt_action = [0] * k

        self.sample_average = sample_average
        self.UCB = UCB
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline

        self.step_idx = 0
        self.step_size = step_size

        self.eps = eps
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
        if self.UCB:
            # At = argmax { Qt(a) + c * sqrt( log(t) / Nt(a) ) }
            ucb_q = self.Qt + self.UCB * np.sqrt( np.log(self.step_idx + 1) / (np.array(self.cnt_action) + 1e-5) )
            best_q = np.max(ucb_q)
            return np.random.choice([i for i, q in enumerate(ucb_q) if q == best_q])
        if self.gradient:
            # probs(a) = exp(H(a)) / sum_b=1_to_k exp(H(b))
            exp_q = np.exp(self.Qt)
            self.q_probs = exp_q / np.sum(exp_q)
            return np.random.choice(range(self.k), p=self.q_probs)
        best_q = np.max(self.Qt)
        return np.random.choice([i for i, q in enumerate(self.Qt) if q == best_q])

    def step(self, action):
        reward = self.get_reward(action)
        self.cnt_action[action] += 1
        if self.sample_average:
            self.Qt[action] = self.Qt[action] + (reward - self.Qt[action]) / self.cnt_action[action] # sample average: new = old + 1/n * (R - old)
        elif self.gradient:
            if self.gradient_baseline:
                baseline = reward
            else:
                baseline = 0
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            # H(t+1) = H(t) + alpha * (R - R_avg) * (1_action - probs(action)) # for taken action
            # H(t+1) = H(t) - alpha * (R - R_avg) * probs(other_action) # for other actions
            self.Qt += self.step_size * (reward - baseline) * (one_hot - self.q_probs)
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

def main():
    # visualize all algorithms
    k = 10
    num_runs = 2000
    all_percent_opt_action = []
    all_avg_reward = []

    algo_params = [
        {"name": "eps=0", "eps": 0, "sample_average": True},
        {"name": "eps=1/32", "eps": 1/32, "sample_average": True},
        {"name": "eps=1/16", "eps": 1/16, "sample_average": True},
        {"name": "eps=1/16 Optimistic", "eps": 1/16, "initial_q": 5.0, "sample_average": True},
        {"name": "UCB c=0.5", "UCB": 0.5, "sample_average": True},
        {"name": "UCB c=1", "UCB": 1, "sample_average": True},
        {"name": "Gradient, baseline", "true_reward": 4.0, "gradient": True, "gradient_baseline": True, "step_size": 0.1},
        {"name": "Gradient, no baseline", "true_reward": 4.0, "gradient": True, "gradient_baseline": False, "step_size": 0.1},
    ]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    plt.figure(figsize=(12, 8))
    for idx, params in enumerate(algo_params):
        print(f"Running simulation for {params['name']}")
        percent_opt_action_list = np.array([0.0] * 1000)
        avg_reward_list = np.array([0.0] * 1000)
        bandit = Bandit(k=k, 
                        eps=params.get("eps", 0.0), 
                        sample_average=params.get("sample_average", True), 
                        UCB=params.get("UCB", None), 
                        gradient=params.get("gradient", False), 
                        gradient_baseline=params.get("gradient_baseline", False), 
                        step_size=params.get("step_size", 0.1))
        for run_idx in tqdm(range(1, num_runs + 1)):
            reward_per_steps, percent_opt_action = run_simulation(bandit)
            avg_reward_list += (reward_per_steps - avg_reward_list) / run_idx # avg over runs
            percent_opt_action_list += (percent_opt_action - percent_opt_action_list) / run_idx # avg over runs
        all_avg_reward.append(avg_reward_list)
        all_percent_opt_action.append(percent_opt_action_list)
        plt.plot(avg_reward_list, label=params['name'], color=colors[idx])


    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.xlim(0, 1000)
    plt.ylim(0, 2.5)
    plt.legend()
    # plt.show()
    plt.savefig("./outputs/all_algos_average_reward.png")

    plt.figure(figsize=(12, 8))
    for idx, params in enumerate(algo_params):
        plt.plot(all_percent_opt_action[idx], label=params['name'], color=colors[idx])
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.xlim(0, 1000)
    plt.ylim(0, 1)
    plt.legend()
    # plt.show()
    plt.savefig("./outputs/all_algos_percent_optimal_action.png")



if __name__ == "__main__":
    # figure22()
    # figure23()
    main()