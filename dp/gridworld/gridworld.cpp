#include <iostream>

using namespace std;
#define pii pair<int, int> 
#define fi first
#define se second
struct State {
    int x;
    int y;
};

pii actionSpace[4] = {pii(0, 1), pii(1, 0), pii(0, -1), pii(-1, 0)};
float actionProb[4] = {0.25, 0.25, 0.25, 0.25}; // uniform rand policy
unsigned int WORLD_SIZE = 4;

bool isTerminal(const State &s) {
    return (s.x == 0 && s.y == 0) || (s.x == WORLD_SIZE-1 && s.y == WORLD_SIZE-1);
}

void compute_state_value(float* state_values) 
{
    float delta = 1e-4;
    float theta = 1e-4;
    float* new_state_value = new float[WORLD_SIZE * WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE * WORLD_SIZE; ++i)
    {
        new_state_value[i] = 0.0;
    }
    int iteration = 0;
    
    float *tmp = NULL;
    do 
    {
        ++iteration;
        delta = 0.0;
        for(int i = 0; i < WORLD_SIZE; ++i)
        {
            for(int j = 0; j < WORLD_SIZE; ++j)
            {
                State s = {i, j};
                float value = 0;
                for(int actionIndex = 0; actionIndex < 4; ++actionIndex)
                {
                    State nextState;
                    float reward = -1.0;
                    nextState = {s.x + actionSpace[actionIndex].fi, s.y + actionSpace[actionIndex].se}; 
                    if(isTerminal(s)) {
                        nextState = s;
                        reward = 0.0;
                    }
                    if(nextState.x < 0 || nextState.x >= WORLD_SIZE || nextState.y < 0 || nextState.y >= WORLD_SIZE) {
                        nextState = s;
                    }
                    // discount = 1.0, actionProb = 0.25
                    value += actionProb[actionIndex] * (reward + 1. * state_values[nextState.x * WORLD_SIZE + nextState.y]);
                    // cout << "  From (" << s.x << ", " << s.y << ") take action " << actionIndex 
                    //      << " to (" << nextState.x << ", " << nextState.y << ") with reward " << reward << endl;
                }
                new_state_value[i * WORLD_SIZE + j] = value;
            }
        }
        for(int i = 0; i < WORLD_SIZE * WORLD_SIZE; ++i)
        {
            if(state_values[i] > new_state_value[i])
                delta = max(delta, state_values[i] - new_state_value[i]);
            else
                delta = max(delta, new_state_value[i] - state_values[i]);
        }
        for (int i = 0; i < WORLD_SIZE * WORLD_SIZE; ++i)
        {
            state_values[i] = new_state_value[i];
        }
    } while (delta >= theta);
    cout << "Converged after " << iteration << " iterations." << endl;
    delete[] new_state_value;
}

int main() {
    float *state_values = new float[WORLD_SIZE * WORLD_SIZE];
    for(int i = 0; i < WORLD_SIZE * WORLD_SIZE; ++i) {
        state_values[i] = 0.0;
    }
    compute_state_value(state_values);
    cout << "State Values: " << endl;
    for(int i = 0; i < WORLD_SIZE; ++i) {
        for(int j = 0; j < WORLD_SIZE; ++j) {
            cout << state_values[i * WORLD_SIZE + j] << "\t";
        }
        cout << endl;
    }
    /*
    State Values: 
0       -13.9989        -19.9984        -21.9982
-13.9989        -17.9986        -19.9984        -19.9984
-19.9984        -19.9984        -17.9986        -13.9989
-21.9982        -19.9984        -13.9989        0
    */
    delete[] state_values;
    return 0;
}