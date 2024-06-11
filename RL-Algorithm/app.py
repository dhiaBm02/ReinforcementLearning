from flask import Flask, jsonify, render_template, request
import json
import random

app = Flask(__name__)

# Environment settings
grid_size = (3, 3)
start = (0, 0)
goal = (2, 2)
obstacles = []  # No obstacles in this simple example
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
num_episodes = 10000
max_steps_per_episode = 200  # To prevent infinite loops

# Initialize Q-table
q_table = {}
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        q_table[(x, y)] = {action: 0.0 for action in actions}


def is_valid_state(state):
    x, y = state
    if (0 <= x < grid_size[0] and 0 <= y < grid_size[1] and state not in obstacles):
        return True
    return False


def get_next_state(state, action):
    delta = action_to_delta[action]
    next_state = (state[0] + delta[0], state[1] + delta[1])
    if is_valid_state(next_state):
        # move_agent_step_by_step(action)
        return next_state
    return state  # If next state is invalid, stay in the current state


def get_reward(state):
    if state == goal:
        return 100  # Reward for reaching the goal
    if state in obstacles:
        return -100  # Penalty for hitting an obstacle
    return 0  # Small penalty for each step to encourage the shortest path


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Exploration
    else:
        max_q = max(q_table[state].values())
        return random.choice([action for action in actions if q_table[state][action] == max_q])  # Exploitation


def save_q_table(filename):
    with open(filename, 'w') as f:
        # Convert dictionary keys to strings for JSON compatibility
        q_table_str_keys = {str(k): v for k, v in q_table.items()}
        json.dump(q_table_str_keys, f)


def load_q_table(filename, forHtml=False):
    with open(filename, 'r') as f:
        q_table_str_keys = json.load(f)

    if forHtml:
        return q_table_str_keys

    # Convert string keys back to tuples
    q_table = {eval(k): v for k, v in q_table_str_keys.items()}
    return q_table


# Training loop


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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train_q_learning():
    # q_table = load_q_table("q_table.json")
    data = request.get_json()
    episodes_to_run = data.get('episodes', 1)
    for episode in range(episodes_to_run):
        state = start
        for _ in range(max_steps_per_episode):  # Limit steps per episode
            action = choose_action(state)
            next_state = get_next_state(state, action)

            if state == next_state:
                continue  # Skip if no valid next state

            reward = get_reward(next_state)

            # Bellman equation update
            max_future_q = max(q_table[next_state].values())
            q_table[state][action] += alpha * (reward + gamma * max_future_q - q_table[state][action])

            print(f"\n{episode=}, {state=}, {next_state=}, {reward=}")
            if episode == 0:
                print(f"\nepisode 0 step {_} Q-Table:")
                for state, actions in q_table.items():
                    print(f"State {state}: {actions}")

            state = next_state

            if state == goal:
                break  # Exit if goal is reached

    save_q_table("q_table.json")
    return jsonify(load_q_table("q_table.json", True))


@app.route('/reset', methods=['POST'])
def reset_q_table():
    global q_table
    q_table = {}
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            q_table[(x, y)] = {action: 0.0 for action in actions}

    save_q_table("q_table.json")
    q_table = load_q_table("q_table.json")
    return load_q_table("q_table.json", True)


if __name__ == '__main__':
    app.run(debug=True)