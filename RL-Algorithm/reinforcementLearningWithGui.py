import random
import json
from pyamaze import maze, agent, COLOR

# Environment settings
grid_size = (10, 10)
start = (9, 9)
goal = (0, 0)
obstacles = []  # No obstacles in this simple example
actions = ['up', 'down', 'left', 'right']
action_to_delta = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Q-learning settings
alpha = 0.9  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Exploration rate
num_episodes = 10000
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
        # move_agent_step_by_step(action)
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
episode0Path = [start]
episode1Path = [start]
episode5000Path = [start]
episode9000Path = [start]


def train_q_learning():
    success_count = 0
    steps_to_goal = []
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

            state = next_state

            if episode == 0:
                episode0Path.append(state)

            if episode == 1:
                episode1Path.append(state)

            if episode == 5000:
                episode5000Path.append(state)

            if episode == 9000:
                episode9000Path.append(state)

            if state == goal:
                print(f"Goal reached in episode {episode} at step {_}")
                success_count = success_count + 1
                steps_to_goal.append(_ + 1)
                break  # Exit if goal is reached

        if episode == 0:
            save_q_table('QTableEpisode0.json')

        if episode == 1:
            save_q_table('QTableEpisode1.json')

        if episode == 5000:
            save_q_table('QTableEpisode5000.json')

        if episode == 9000:
            save_q_table('QTableEpisode9000.json')

    success_rate = success_count / num_episodes
    avg_steps_to_goal = sum(steps_to_goal) / len(steps_to_goal)
    print(success_count, len(steps_to_goal))
    print(f"Success rate: {success_rate:.2f}, Average steps to goal: {avg_steps_to_goal:.2f}")


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
            continue
        path.append(next_state)
        state = next_state
    return path


def create_open_maze_with_boundaries(rows, cols):
    m = maze(rows, cols)

    # Set all internal walls to be open
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            m.maze_map[(r, c)] = {'N': 1, 'S': 1, 'E': 1, 'W': 1}

    # Set the boundary walls to remain intact
    for r in range(1, rows + 1):
        m.maze_map[(r, 1)]['W'] = 0  # Left boundary
        m.maze_map[(r, cols)]['E'] = 0  # Right boundary
    for c in range(1, cols + 1):
        m.maze_map[(1, c)]['N'] = 0  # Top boundary
        m.maze_map[(rows, c)]['S'] = 0  # Bottom boundary

    return m


# def move_agent_step_by_step(move):
#     if move == 'up':
#         a.moveUp(None)
#     elif move == 'down':
#         a.moveDown(None)
#     elif move == 'left':
#         a.moveLeft(None)
#     elif move == 'right':
#         a.moveRight(None)

#m.run()


# Main logic
if __name__ == "__main__":
    # set up the maze
    m = create_open_maze_with_boundaries(10, 10)
    m.CreateMaze()

    a0 = agent(m, color=COLOR.green, filled=True, footprints=True)
    a1 = agent(m, color=COLOR.red, filled=True, footprints=True)

    a5000 = agent(m, color=COLOR.blue, filled=True, footprints=True)

    a9000 = agent(m, color=COLOR.light, filled=True, footprints=True)

    a = agent(m, color=COLOR.yellow, filled=True, footprints=True)

    # Train the model and save the Q-table
    train_q_learning()
    save_q_table('QTable.json')

    # Load the Q-table
    q_table = load_q_table('QTable.json')

    # Print the best path found
    best_path = extract_path(start, goal)
    print("Best path found:", best_path)

    # Print the final Q-Table for verification
    print("\nFinal Q-Table:")
    for state, actions in q_table.items():
        print(f"State {state}: {actions}")

    converted_path = [(x + 1, y + 1) for (x, y) in best_path]
    # print(m.maze_map)

    converted_episode0Path = [(x + 1, y + 1) for (x, y) in episode0Path]
    m.tracePath({a0: converted_episode0Path})

    converted_episode1Path = [(x + 1, y + 1) for (x, y) in episode1Path]
    m.tracePath({a1: converted_episode1Path})

    converted_episode5000Path = [(x + 1, y + 1) for (x, y) in episode5000Path]
    m.tracePath({a5000: converted_episode5000Path})

    converted_episode9000Path = [(x + 1, y + 1) for (x, y) in episode9000Path]
    m.tracePath({a9000: converted_episode9000Path})

    m.tracePath({a: converted_path})

    m.run()
