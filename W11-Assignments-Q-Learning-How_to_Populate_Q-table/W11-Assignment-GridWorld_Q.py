# ("NON-DETERMINISTIC, Q-LEARNING")
# Aspect            | Description
# Movement          | Non-deterministic (intended move can randomly become left/right sometimes).
# Learning          | Q-Learning (learns value of taking an action at a state: Q(s,a)).
# Rewards           | Reward assigned to state-action pairs.
# Policy            | Chooses the best action at each state based on Q-values.
# State-Action Info | Each (state, action) pair has a separate Q-value.
# Update Formula    | Q-Learning rule: Q(s,a)=Q(s,a)+lr×(γ×reward_next−Q(s,a))
# | Exploration | Also random actions 30% of the time. | | Backpropagation | Updates the state-action pair values after each game. |

import numpy as np
# Board size: 3 rows × 4 columns.
# Winning position: (row 0, col 3).
# Losing position: (row 1, col 3).
# Starting point: (row 2, col 0).
# DETERMINISTIC = False → movement is non-deterministic (randomness involved).
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = False


class State:
    def __init__(self, state=START):#Creates a new board (game environment) and sets starting position.
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS]) #Initializes a 3×4 grid filled with 0s.
        self.board[1, 1] = -1 #Block the cell (1,1) with -1 → Wall/Obstacle
        self.state = state #Set current position.
        self.isEnd = False #Mark game not ended.
        self.determine = DETERMINISTIC #Set whether moves are deterministic or not.

    def giveReward(self):
        # Returns reward depending on current position:
        # +1 if Win,
        # -1 if Lose,
        # 0 otherwise.
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def isEndFunc(self):
        # Checks if the current state is WIN or LOSE, and sets isEnd = True.
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def _chooseActionProb(self, action):
        # Introduces non-deterministic behavior:
        # 80% chance move in intended direction,
        # 10% chance to go left,
        # 10% chance to go right.
        # Example: if you intend to go "up", you might randomly instead go "left" or "right" sometimes

        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        # Computes the next position based on action:
        #  If deterministic, simple move (up, down, left, right).
        #  If non-deterministic:
        #     Choose an actual move using _chooseActionProb.
        #     Then call nxtPosition again recursively.
        # After choosing the action:
        #  Check boundaries (can't go off the board).
        #  Check for walls (can't go into (1,1)).
        #  If invalid move →  stay at the current state.
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState = self.nxtPosition(action)

        # if next state is legal
        if (nxtState[0] >= 0) and (nxtState[0] <= 2):
            if (nxtState[1] >= 0) and (nxtState[1] <= 3):
                if nxtState != (1, 1):
                    return nxtState
        return self.state

    def showBoard(self):
        # Displays the board visually:
        # * = Agent,
        # z = Wall,
        # 0 = Empty space.
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


class Agent:

    def __init__(self): #Sets up the agent.
        self.states = []  # Empty List to record (state, action) pairs during the game.
        self.actions = ["up", "down", "left", "right"] #List of possible actions.
        self.State = State() #Create an instance of State (i.e., the environment).
        self.isEnd = self.State.isEnd
        self.lr = 0.2 # lr = learning rate (how fast to update Q-values),
        self.exp_rate = 0.3 # exp_rate = exploration rate (30% random actions)
        self.decay_gamma = 0.9 # decay_gamma = discount factor for future rewards (how much to value rewards later).

        # initial Q values
        self.Q_values = {}
        # Initializes Q-values:
        # For every (i,j) position,
        # For each possible action,
        # Set Q-value to 0 initially.
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        # Chooses an action:
        #   Random action (with 30% probability) for exploration.
        #   Greedy action (choosing the highest Q-value) otherwise.
        # (If multiple actions have same Q-value, picks the last one checked.)
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action): #Takes the chosen action and returns the new State after the move.
        position = self.State.nxtPosition(action)
        # update State
        return State(state=position)

    def reset(self): #Resets the agent back to the starting point and clears history after each episode.
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd

    def play(self, rounds=10):
        # Main loop to train the agent.
        i = 0
        while i < rounds:
            # For a number of rounds (games):
            # to the end of game back propagate reward
            # When game ends:
            # If agent has reached WIN/LOSE state:
            #   Get the reward.
            #   Set Q-values of end state for all actions to that reward.
            if self.State.isEnd:
                # back propagate
                # Then update the recorded state-action pairs in reverse (backpropagation):
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print("Game End Reward", reward)
                # This uses the Q-learning update rule:
                # Q(s,a)=Q(s,a)+lr×(γ×reward_next−Q(s,a))
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1
            else: # When game not ended:
                # Choose an action and record current (state, action).
                action = self.chooseAction()
                # append trace
                self.states.append([(self.State.state), action])
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                # Move to the next state.
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc() #Check if now at an end state (win/lose).
                print("nxt state", self.State.state)
                print("---------------------")
                self.isEnd = self.State.isEnd

# Main Execution
# Create the agent.
# Print initial Q-values (all zeros).
# Play 50 games to learn.
# Print the updated Q-values (agent’s learned strategy).
if __name__ == "__main__":
    ag = Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)

    ag.play(50)
    print("latest Q-values ... \n")
    print(ag.Q_values)


#  Main Concepts Used
# Concept                      | Purpose
# Non-deterministic actions    | Simulate real-world uncertainty.
# Q-learning                   | Learn best action-values from experience.
# Exploration vs Exploitation  | Random moves vs greedy best moves.
# Backpropagation              | Update the agent's Q-values after each game.

# This agent learns how to reach WIN and avoid LOSE even when actions are not fully reliable.
#
# It stores a Q-table: how good it is to take each action at each board position.

# Feature              | GridWorld.py(State Value Learning)       | Second Code - GridWorld_Q.py (Q-Learning)
# Type of movement     | Deterministic                            | Non-deterministic (80% intended, 20% random)
# Learning target      | State values V(s)                        | State-Action values Q(s,a)
# Focus                | Find best next state                     | Find best action to take
# Complexity           | Simpler                                  | More detailed (handles randomness better)
# Algorithm            | TD(0) Learning                           | Q-Learning
# When random effects | Only at action choosing (exploration)     | At movement itself (environment is noisy)

# First code teaches:
# ➔ "Which position is good to reach?"
#
# Second code teaches:
# ➔ "Which action should I take at this position, given randomness?"