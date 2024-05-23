import random
import json
import webbrowser

# Environment settings
grid_size = (5, 5)
start = (0, 0)
goal = (4, 4)
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
num_episodes = 1000
max_steps_per_episode = 100  # To prevent infinite loops

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
        return next_state
    return state  # If next state is invalid, stay in the current state


def get_reward(state):
    if state == goal:
        return 100  # Reward for reaching the goal
    if state in obstacles:
        return -100  # Penalty for hitting an obstacle
    return -1  # Small penalty for each step to encourage the shortest path


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


def load_q_table(filename):
    with open(filename, 'r') as f:
        q_table_str_keys = json.load(f)
    # Convert string keys back to tuples
    q_table = {eval(k): v for k, v in q_table_str_keys.items()}
    return q_table


# Training loop
def train_q_learning():
    for episode in range(num_episodes):
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


def save_q_table_html(q_table, filename):
    q_table_str_keys = {str(k): v for k, v in q_table.items()}

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Q-Learning Visualization</title>
        <style>
            table {{
                border-collapse: collapse;
                margin: 20px;
            }}
            th, td {{
                border: 1px solid black;
                padding: 10px;
                text-align: center;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat({grid_size[1]}, 40px);
                grid-template-rows: repeat({grid_size[0]}, 40px);
                gap: 1px;
                margin: 20px;
            }}
            .cell {{
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid black;
                background-color: lightgrey;
            }}
            .start {{
                background-color: green;
            }}
            .goal {{
                background-color: gold;
            }}
            .obstacle {{
                background-color: black;
            }}
        </style>
    </head>
    <body>
        <h1>Q-Learning Visualization</h1>
        <h2>Grid</h2>
        <div class="grid">
    """

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            cell_class = 'cell'
            if (x, y) == start:
                cell_class += ' start'
            elif (x, y) == goal:
                cell_class += ' goal'
            elif (x, y) in obstacles:
                cell_class += ' obstacle'
            html_content += f'<div class="{cell_class}">({x},{y})</div>'

    html_content += """
        </div>
        <h2>Q-Table</h2>
        <table>
            <tr>
                <th>State</th>
                <th>UP:Q-Value</th>
                <th>DOWN:Q-Value</th>
                <th>LEFT:Q-Value</th>
                <th>RIGHT:Q-Value</th>
            </tr>
    """

    for state, actions in q_table_str_keys.items():
        html_content += f"""
                    <tr>
                        <td>{state}</td>
                    """
        for action, value in actions.items():
            html_content += f"""
                <td>{action}:{value:.2f}</td>
            """

        html_content += f"""
                    </tr>
                    """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(filename, 'w') as f:
        f.write(html_content)


# Main logic
if __name__ == "__main__":
    # Train the model and save the Q-table
    train_q_learning()
    save_q_table('QTable.json')

    # Load the Q-table
    q_table = load_q_table('QTable.json')

    # Save the Q-table as an HTML file for visualization
    save_q_table_html(q_table, 'QTable.html')

    webbrowser.open('QTable.html')

    # Print the best path found
    best_path = extract_path(start, goal)
    print("Best path found:", best_path)

    # Print the final Q-Table for verification
    print("\nFinal Q-Table:")
    for state, actions in q_table.items():
        print(f"State {state}: {actions}")
