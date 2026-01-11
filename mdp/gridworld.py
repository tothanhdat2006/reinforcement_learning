import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

DISCOUNT = 0.9
A_POS = (0, 1)
B_POS = (0, 3)
AA_POS = (4, 1)
BB_POS = (2, 3)
A_REWARD = 10
B_REWARD = 5
C = 0

action_value = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1)   # left
}

def step(state: tuple, action: int):
    if state == A_POS:
        return AA_POS, A_REWARD
    if state == B_POS:
        return BB_POS, B_REWARD
    
    i, j = state
    if i == 0 and action == 0:
        return (i, j), -1
    if i == 4 and action == 2:
        return (i, j), -1
    if j == 0 and action == 3:
        return (i, j), -1
    if j == 4 and action == 1:
        return (i, j), -1
    di, dj = action_value[action]
    new_state = (i + di, j + dj)
    return new_state, C

def draw_world(grid, name="./outputs/fig32.png", title="State Value Function discount=0.9"):
    plt.figure(figsize=(10,10))
    plt.imshow(grid, cmap='rainbow', interpolation='nearest')
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{grid[i, j]:.1f}", ha='center', va='center', color='black')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.savefig(name)

def draw_policy(grid, name="./outputs/fig35_policy.png", title="Optimal Policy discount=0.9"):
    policy_grid = np.empty((5, 5), dtype=object)
    action_symbols = {
        0: '↑',
        1: '→',
        2: '↓',
        3: '←'
    }
    for i in range(5):
        for j in range(5):
            values = []
            for action in range(len(action_value)):
                (next_i, next_j), reward = step((i, j), action)
                values.append(reward + DISCOUNT * grid[next_i, next_j])
            best_actions = np.argwhere(values == np.max(values)).flatten()
            policy_grid[i, j] = [action_symbols[a] for a in best_actions]

    
    plt.figure(figsize=(10,10))
    tb = Table(plt.gca(), bbox=[0,0,1,1])
    nrows, ncols = policy_grid.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    
    # show all optimal actions
    for i in range(nrows):
        for j in range(ncols):
            tb.add_cell(i, j, width, height, text=policy_grid[i, j], loc='center', edgecolor='black', facecolor='white')

    for i in range(nrows):
        tb.add_cell(i, -1, width, height, text=i, loc='right', edgecolor='none', facecolor='none')
    for j in range(ncols):
        tb.add_cell(-1, j, width, height/2, text=j, loc='center', edgecolor='none', facecolor='none')
    plt.gca().add_table(tb)
    # plt.title(title)
    plt.axis('off')
    plt.savefig(name)


def fig32():
    grid = np.zeros((5, 5))
    num_it = 0
    while True: # T < inf 
        num_it += 1
        print("Iteration", num_it)
        grid_new = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                for action in range(len(action_value)):
                    (next_i, next_j), reward = step((i, j), action)
                    # Bellman equation: v_pi_(s) = sum_a pi(a|s) * sum_{s', r} p(s', r|s, a) [ r + gamma * v_pi_(s') ]
                    grid_new[i, j] += 1/(len(action_value)) * (reward + DISCOUNT * grid[next_i, next_j]) 
        if np.sum(np.abs(grid - grid_new)) < 1e-4:
            # convergence
            # print(grid_new)
            draw_world(grid_new)
            break
        grid = grid_new

def fig35():
    grid = np.zeros((5, 5))
    num_it = 0
    while True: # T < inf 
        num_it += 1
        print("Iteration", num_it)
        grid_new = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                values = []
                for action in range(len(action_value)):
                    (next_i, next_j), reward = step((i, j), action)
                    # Bellman equation: v_pi_(s) = sum_a pi(a|s) * sum_{s', r} p(s', r|s, a) [ r + gamma * v_pi_(s') ]
                    values.append(reward + DISCOUNT * grid[next_i, next_j])
                grid_new[i, j] = max(values)
        if np.sum(np.abs(grid - grid_new)) < 1e-4:
            # convergence
            # print(grid_new)
            draw_world(grid_new, name="./outputs/fig35.png", title="Optimal State Value Function discount=0.9")
            draw_policy(grid_new, name="./outputs/fig35_policy.png", title="Optimal Policy discount=0.9")
            break
        grid = grid_new

if __name__ == "__main__":
    fig32()
    fig35()