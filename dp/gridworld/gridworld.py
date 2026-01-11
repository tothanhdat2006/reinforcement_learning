import numpy as np

WORLD_SIZE = 4

def is_terminal(state: tuple) -> bool:
    return state in [(0, 0), (WORLD_SIZE-1, WORLD_SIZE-1)]

def step(state: tuple, action: int):
    if is_terminal(state):
        return state, 0 # nothing happens in terminal state
    
    # action set
    action_value = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1)   # left
    }
    # off grid
    x, y = state
    new_x, new_y = x + action_value[action][0], y + action_value[action][1]
    if new_x < 0 or new_x > WORLD_SIZE-1 or new_y < 0 or new_y > WORLD_SIZE-1:
        return state, -1    
    return (new_x, new_y), -1

 
def compute_state_value():
    # V = np.random.rand(WORLD_SIZE, WORLD_SIZE)
    V = np.zeros((WORLD_SIZE, WORLD_SIZE))
    theta = 1e-4
    while True:
        delta = 0
        new_state_value = V.copy()
        # Loop for each s in S
        for (i, j) in [(i, j) for i in range(WORLD_SIZE) for j in range(WORLD_SIZE)]:
            v = 0
            for action in range(4):
                (next_i, next_j), reward = step((i, j), action)
                # uniform random policy & episodic with no discounting
                v += 0.25 * (reward + 1. * V[next_i, next_j])
            new_state_value[i, j] = v
        
        delta = abs(V - new_state_value).max()
        if delta < theta:
            break
        V = new_state_value
    return V

def compute_action_value():
    Q = np.zeros((WORLD_SIZE, WORLD_SIZE, 4)) # Q(s, a)
    theta = 1e-4
    while True:
        delta = 0
        new_action_value = Q.copy()
        # Loop for each s in S
        for (i, j) in [(i, j) for i in range(WORLD_SIZE) for j in range(WORLD_SIZE)]:
            q = 0
            # Q_{k+1} (s, a) = sum_{s', r} p(s', r | s, a) * [r + sum_{a'} policy(a' | s') Q_{k} (s', a') * discount]
        
        delta = abs(V - new_action_value).max()
        if delta < theta:
            break
        Q = new_action_value
    return V

if __name__ == "__main__":
    V = compute_state_value()
    print(V)
    """ 
0       -13.9989        -19.9984        -21.9982
-13.9989        -17.9986        -19.9984        -19.9984
-19.9984        -19.9984        -17.9986        -13.9989
-21.9982        -19.9984        -13.9989        0
    """