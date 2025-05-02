import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Grid world settings
BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
JUMP_FROM = (1, 3)
JUMP_TO = (3, 3)
START = (1, 0)
OBSTACLES = [(2, 2), (2, 3), (2, 4), (3, 2)]

DETERMINISTIC = False

# Rewards
REWARD_WIN = 10
REWARD_JUMP = 5
REWARD_DEFAULT = -1

# Actions: 1-North, 2-South, 3-East, 4-West
ACTIONS = {
    "1": (-1, 0),
    "2": (1, 0),
    "3": (0, 1),
    "4": (0, -1)
}

ACTION_NAMES = {
    "1": "North",
    "2": "South",
    "3": "East",
    "4": "West"
}

DIRECTION_SYMBOLS = {
    "1": "↑",
    "2": "↓",
    "3": "→",
    "4": "←"
}

class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        for (i, j) in OBSTACLES:
            self.board[i, j] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == WIN_STATE:
            return REWARD_WIN
        else:
            return REWARD_DEFAULT

    def isEndFunc(self):
        if self.state == WIN_STATE:
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "1":
            return np.random.choice(["1", "4", "3"], p=[0.8, 0.1, 0.1])
        if action == "2":
            return np.random.choice(["2", "4", "3"], p=[0.8, 0.1, 0.1])
        if action == "3":
            return np.random.choice(["3", "1", "2"], p=[0.8, 0.1, 0.1])
        if action == "4":
            return np.random.choice(["4", "1", "2"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        if not self.determine:
            action = self._chooseActionProb(action)
            self.determine = True
            return self.nxtPosition(action)
        else:
            dr, dc = ACTIONS[action]
            r, c = self.state
            next_r, next_c = r + dr, c + dc

            if next_r < 0 or next_r >= BOARD_ROWS or next_c < 0 or next_c >= BOARD_COLS:
                return self.state

            if (next_r, next_c) in OBSTACLES:
                return self.state

            if (next_r, next_c) == JUMP_FROM:
                return JUMP_TO

            return (next_r, next_c)


class Agent:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.states = []
        self.actions = ["1", "2", "3", "4"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 1.0
        self.exp_rate = 0.3
        self.decay_gamma = 0.9
        self.cumulative_rewards = []

        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if (i, j) not in OBSTACLES:
                    self.Q_values[(i, j)] = {a: 0.0 for a in self.actions}


    def chooseAction(self):
        if np.random.uniform(0, 1) <= self.exp_rate:
            return np.random.choice(self.actions)
        else:
            mx_nxt_reward = float('-inf')
            action = np.random.choice(self.actions)
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        next_pos = self.State.nxtPosition(action)
        self.states.append((self.State.state, action))
        reward = REWARD_DEFAULT ## Always step cost of -1 first
        if next_pos == WIN_STATE:
            reward = REWARD_WIN
        elif next_pos == JUMP_TO:
            reward = REWARD_JUMP
        self.State.state = next_pos
        self.State.isEndFunc()
        return reward


    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd

    def play(self, rounds):
        last_30_rewards = []
        for episode in range(rounds):
            self.reset()
            cumulative_reward = 0
            while not self.State.isEnd:
                current_state = self.State.state
                action = self.chooseAction()
                reward = self.takeAction(action)
                cumulative_reward += reward

                next_state = self.State.state
                current_q = self.Q_values[current_state][action]
                max_future_q = max(self.Q_values[next_state].values()) if next_state in self.Q_values else 0.0

                # Q-learning
                new_q = current_q + self.lr * (reward + self.decay_gamma * max_future_q - current_q)
                self.Q_values[current_state][action] = round(new_q, 3)

            self.cumulative_rewards.append(cumulative_reward)
            last_30_rewards.append(cumulative_reward)

            if len(last_30_rewards) > 30:
                last_30_rewards.pop(0)

            if len(last_30_rewards) == 30 and np.mean(last_30_rewards) >= 10.0:
                print(f"Stopping early at Episode {episode + 1}, Avg Reward={np.mean(last_30_rewards):.2f}")
                break
            else:
                action = self.chooseAction()
                reward = self.takeAction(action)
                cumulative_reward += reward # accumulate all rewards, step + jump/win
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd

        self.state_values = {
            (i, j): max(self.Q_values[(i, j)].values())
            for (i, j) in self.Q_values
        }

    def showbestPolicy(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if (i, j) in OBSTACLES:
                    out += "###".ljust(6) + ' | '
                else:
                    best_action = max(self.Q_values[(i, j)], key=self.Q_values[(i, j)].get)
                    out += ACTION_NAMES[best_action].ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def showStateValues(self): #Prints the learned values of all positions in a grid form.

        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if (i, j) in OBSTACLES:
                    out += "###".ljust(6) + ' | '
                else:
                    out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def showQValues(self): #Prints the learned values of all positions in a grid form.
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = ' | '
            for j in range(0, BOARD_COLS):
                if (i, j) in OBSTACLES:
                    out += "###".ljust(6) + ' | '
                else:
                    out += str(self.Q_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def plot_rewards(self):
        plt.plot(self.cumulative_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Training Progress')
        plt.grid()
        plt.show()

    def plot_state_values(self):
        values = np.zeros((BOARD_ROWS, BOARD_COLS))
        for (i, j) in self.Q_values:
            values[i, j] = max(self.Q_values[(i, j)].values())

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, BOARD_COLS)
        ax.set_ylim(0, BOARD_ROWS)
        ax.set_xticks(np.arange(0, BOARD_COLS+1))
        ax.set_yticks(np.arange(0, BOARD_ROWS+1))
        ax.grid(True)

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if (i, j) in OBSTACLES:
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, color='black'))
                elif (i, j) == WIN_STATE:
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, color='cyan'))
                    ax.text(j+0.5, i+0.5, '+10', ha='center', va='center')
                elif (i, j) == JUMP_FROM:
                    ax.text(j+0.5, i+0.5, '+5', ha='center', va='center', color='blue')
                else:
                    ax.text(j+0.5, i+0.5, f"{values[i,j]:.1f}", ha='center', va='center')

        plt.gca().invert_yaxis()
        plt.title("State Value Visualization")
        plt.show()

    def plot_policy(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, BOARD_COLS)
        ax.set_ylim(0, BOARD_ROWS)
        ax.set_xticks(np.arange(0, BOARD_COLS+1))
        ax.set_yticks(np.arange(0, BOARD_ROWS+1))
        ax.grid(True)

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if (i, j) in OBSTACLES:
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, color='black'))
                elif (i, j) == WIN_STATE:
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, color='cyan'))
                    ax.text(j+0.5, i+0.5, 'WIN', ha='center', va='center')
                elif (i, j) == JUMP_FROM:
                    ax.text(j+0.5, i+0.5, 'Jump', ha='center', va='center', color='blue')
                elif (i, j) in self.Q_values:
                    best_action = max(self.Q_values[(i, j)], key=self.Q_values[(i, j)].get)
                    ax.text(j+0.5, i+0.5, DIRECTION_SYMBOLS[best_action], ha='center', va='center', fontsize=14)

        plt.gca().invert_yaxis()
        plt.title("Policy Visualization")
        plt.show()


# Main Execution
if __name__ == "__main__":
    ag = Agent()
    print("Initial Q-values:")
    print(ag.Q_values)

    ag.play(100)
    print("\nLearned Best Actions (Policy):")
    ag.showbestPolicy()

    print("\nState Values Visualization:")
    ag.showStateValues()

    print("\nQ-Values Visualization:")
    ag.showQValues()

    ag.plot_rewards()
    ag.plot_policy()
    ag.plot_state_values()


