import numpy as np

NUM_STATES = 200
DISCOUNT = 1

def generate_episode() -> tuple:
    pass

def first_visit_MC(policy) -> np.ndarray:
    V = np.zeros(NUM_STATES)
    Returns = np.empty(shape=(NUM_STATES))
    while True:
        # print("Episode")
        S, A, R = generate_episode() # TODO
        T = len(S)
        G = 0
        for t in range(T-1, 0, -1): 
            G = DISCOUNT * G + R[t+1]
            if S[t] not in S[:t]:
                Returns[S[t]].append(G)
                V[S[t]] = np.average(Returns[S[t]])
        if True: # TODO
            break

    return Returns

def main():
    pass

if __name__ == "__main__":
    main()