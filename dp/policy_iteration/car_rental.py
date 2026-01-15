import numpy as np

WORLD_SIZE = 20

def is_terminal(state: tuple) -> bool:
    return state in [(0, 0), (WORLD_SIZE-1, WORLD_SIZE-1)]

def policy_eval(V, policy):
    pass

def policy_improve(V, policy):
    pass

def main():
    pass

if __name__ == "__main__":
    main()