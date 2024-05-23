import random
import json

# Environment settings
grid_size = (5, 5)
start_positions = [(0, 0), (4, 4), (0, 4), (4, 0)]
goals = [(2, 2), (3, 3), (1, 4)]
actions = ['up', 'down', 'left', 'right']
action_to_delta = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Q-learning settings
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
num_episodes = 5000
max_steps_per_episode = 200  # To prevent infinite loops

# Initialize Q-table
q_table = {}
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        q_table[(x, y)] = {action: 0.0 for action in actions}


def is_valid_state(state):
    x, y = state
    return 0 <= x < grid_size[0] and 0 <= y < grid_size[1]


def get_next_state(state, action):
    delta = action_to_delta[action]
    next_state = (state[0] + delta[0], state[1] + delta[1])
    return next_state if is_valid_state(next_state) else state


def get_reward(state, goal):
    if state == goal:
        return 100  # Reward for reaching the goal
    return -1  # Small penalty for each step to encourage shortest path


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Exploration
    else:
        max_q = max(q_table[state].values())
        return random.choice([action for action in actions if q_table[state][action] == max_q])  # Exploitation


def save_q_table(filename, q_table):
    with open(filename, 'w') as f:
        q_table_str_keys = {str(k): v for k, v in q_table.items()}
        json.dump(q_table_str_keys, f)


def load_q_table(filename):
    with open(filename, 'r') as f:
        q_table_str_keys = json.load(f)
    q_table = {eval(k): v for k, v in q_table_str_keys.items()}
    return q_table


# Training loop
def train_q_learning():
    for episode in range(num_episodes):
        state = random.choice(start_positions)
        goal = random.choice(goals)
        for _ in range(max_steps_per_episode):  # Limit steps per episode
            action = choose_action(state)
            next_state = get_next_state(state, action)
            reward = get_reward(next_state, goal)

            # Bellman equation update
            max_future_q = max(q_table[next_state].values())
            q_table[state][action] += alpha * (reward + gamma * max_future_q - q_table[state][action])

            state = next_state

            if state == goal:
                break  # Exit if goal is reached


# Extract the best path
def extract_path(start, goal):
    path = [start]
    state = start
    for _ in range(max_steps_per_episode):  # Limit steps to avoid infinite loops
        if state == goal:
            break
        action = max(q_table[state], key=q_table[state].get)
        next_state = get_next_state(state, action)
        if next_state == state:  # Prevent infinite loop
            break
        path.append(next_state)
        state = next_state
    return path


# Main logic
if __name__ == "__main__":
    # Train the model and save the Q-table
    # train_q_learning()
    # save_q_table('q_table.json', q_table)

    # Load the Q-table
    q_table = load_q_table('q_table.json')

    # Print the best path found for each start position and each goal
    # for goal in goals:
    #    print(f"\nPaths to goal {goal}:")
    #    for start in start_positions:
    #        best_path = extract_path(start, goal)
    #        print(f"  Best path from {start} to {goal}: {best_path}")

    # Print the best path found for a specific start and goal
    _start = (1, 0)
    _goal = (4, 2)
    best_path = extract_path(_start, _goal)
    print(f"  Best path from {_start} to {_goal}: {best_path}")

    # Print the final Q-Table for verification
    print("\nFinal Q-Table:")
    for state, actions in q_table.items():
        print(f"State {state}: {actions}")
