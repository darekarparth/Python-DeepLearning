import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# -------------------
# ENVIRONMENT SETUP
# -------------------

# Grid size
ROWS, COLS = 5, 5

# States
START = (1, 0)  # 2nd row, 1st column
WIN_STATE = (4, 4)
JUMP_FROM = (1, 3)
JUMP_TO = (3, 3)
OBSTACLES = {(2, 2), (2, 3), (2, 4), (3, 2)}

# Rewards
REWARD_WIN = 10
REWARD_JUMP = 5
REWARD_DEFAULT = -1

# Actions: North, South, East, West
ACTIONS = {
    "1": (-1, 0),  # North
    "2": (1, 0),  # South
    "3": (0, 1),  # East
    "4": (0, -1)  # West
}
ACTION_NAMES = {"1": "↑", "2": "↓", "3": "→", "4": "←"}


# -------------------
# AGENT DEFINITION
# -------------------

class QLearningAgent:
    def __init__(self, alpha=1.0, gamma=0.9, epsilon=0.3):
        self.q_table = {(i, j): {a: 0.0 for a in ACTIONS.keys()} for i in range(ROWS) for j in range(COLS) if
                        (i, j) not in OBSTACLES}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state = START

    def reset(self):
        self.state = START

    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(ACTIONS.keys()))  # Explore
        else:
            state_q = self.q_table[self.state]
            return max(state_q, key=state_q.get)  # Exploit

    def step(self, action):
        r, c = self.state
        dr, dc = ACTIONS[action]
        next_r, next_c = r + dr, c + dc

        # Boundary conditions
        next_r = min(max(0, next_r), ROWS - 1)
        next_c = min(max(0, next_c), COLS - 1)
        next_state = (next_r, next_c)

        # Obstacle check
        if next_state in OBSTACLES:
            next_state = self.state  # Blocked, stay in place

        # Jump logic
        if next_state == JUMP_FROM:
            reward = REWARD_JUMP
            next_state = JUMP_TO
        elif next_state == WIN_STATE:
            reward = REWARD_WIN
        else:
            reward = REWARD_DEFAULT

        return next_state, reward

    def update_q(self, action, reward, next_state):
        current_q = self.q_table[self.state][action]
        next_max_q = max(self.q_table[next_state].values()) if next_state in self.q_table else 0.0
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[self.state][action] = new_q

    def take_action(self, action):
        next_state, reward = self.step(action)
        self.update_q(action, reward, next_state)
        self.state = next_state
        return reward, next_state


# -------------------
# TRAINING FUNCTION
# -------------------

def train_agent(agent, max_episodes=1000, check_episodes=30, stop_avg_reward=10):
    episode_rewards = []
    recent_rewards = []

    for episode in range(max_episodes):
        agent.reset()
        cumulative_reward = 0

        while True:
            action = agent.choose_action()
            reward, next_state = agent.take_action(action)
            cumulative_reward += reward

            if next_state == WIN_STATE or next_state == JUMP_TO:
                break  # End of episode

        episode_rewards.append(cumulative_reward)
        recent_rewards.append(cumulative_reward)

        # Maintain last N rewards
        if len(recent_rewards) > check_episodes:
            recent_rewards.pop(0)

        # Check stopping condition
        if len(recent_rewards) == check_episodes:
            avg_reward = sum(recent_rewards) / check_episodes
            if avg_reward >= stop_avg_reward:
                print(f"Early stopping at episode {episode + 1} (Avg reward = {avg_reward:.2f})")
                break

    return episode_rewards


# -------------------
# VISUALIZATION FUNCTIONS
# -------------------

def visualize_state_values(values):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks(np.arange(0, COLS + 1, 1))
    ax.set_yticks(np.arange(0, ROWS + 1, 1))
    ax.grid(True)

    for i in range(ROWS):
        for j in range(COLS):
            if (i, j) in OBSTACLES:
                ax.add_patch(patches.Rectangle((j, ROWS - 1 - i), 1, 1, color='black'))
            elif (i, j) == WIN_STATE:
                ax.add_patch(patches.Rectangle((j, ROWS - 1 - i), 1, 1, color='cyan'))
                ax.text(j + 0.5, ROWS - 1 - i + 0.5, "+10", ha='center', va='center', fontsize=12)
            elif (i, j) == JUMP_FROM:
                ax.text(j + 0.5, ROWS - 1 - i + 0.5, "Jump", ha='center', va='center', fontsize=10, color='blue')
            else:
                ax.text(j + 0.5, ROWS - 1 - i + 0.5, f"{values[i, j]:.1f}", ha='center', va='center', fontsize=10)

    plt.title("State Value Visualization")
    plt.gca().invert_yaxis()
    plt.show()


def visualize_policy(agent):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks(np.arange(0, COLS + 1, 1))
    ax.set_yticks(np.arange(0, ROWS + 1, 1))
    ax.grid(True)

    for i in range(ROWS):
        for j in range(COLS):
            if (i, j) in OBSTACLES:
                ax.add_patch(patches.Rectangle((j, ROWS - 1 - i), 1, 1, color='black'))
            elif (i, j) == WIN_STATE:
                ax.add_patch(patches.Rectangle((j, ROWS - 1 - i), 1, 1, color='cyan'))
                ax.text(j + 0.5, ROWS - 1 - i + 0.5, "WIN", ha='center', va='center', fontsize=10)
            elif (i, j) == JUMP_FROM:
                ax.text(j + 0.5, ROWS - 1 - i + 0.5, "Jump", ha='center', va='center', fontsize=8, color='blue')
            elif (i, j) in agent.q_table:
                best_action = max(agent.q_table[(i, j)], key=agent.q_table[(i, j)].get)
                ax.text(j + 0.5, ROWS - 1 - i + 0.5, ACTION_NAMES[best_action], ha='center', va='center', fontsize=16)

    plt.title("Learned Policy (Best Actions)")
    plt.gca().invert_yaxis()
    plt.show()


# -------------------
# RUNNING EVERYTHING
# -------------------

if __name__ == "__main__":
    agent = QLearningAgent(alpha=1.0, gamma=0.9, epsilon=0.3)
    rewards = train_agent(agent, max_episodes=1000, check_episodes=30, stop_avg_reward=10)

    # Plot reward curve
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Progress')
    plt.grid()
    plt.show()

    # Visualize final learned state values
    final_state_values = np.zeros((ROWS, COLS))
    for (i, j) in agent.q_table:
        final_state_values[i, j] = max(agent.q_table[(i, j)].values())
    visualize_state_values(final_state_values)

    # Visualize best policy
    visualize_policy(agent)
