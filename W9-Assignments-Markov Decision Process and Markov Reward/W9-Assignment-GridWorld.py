'''
The goals of this practice are to have a clear understanding of some key concepts about
reinforcement learning and Markov Decision Process. You need to research by reading
recommended books or web materials before you answer the questions. You also need to
exercise on answering the questions in E1.2

1) Read Reinforcement Learning – Implement Grid World at
https://towardsdatascience.com/reinforcement-learning-implement-grid-world
from-scratch-c5963765ebff
2) Download the code and create a Python file with the name of “grid-world” using
PyCharm, cut and past the code into the grid-world
3) Run the rid-world
4) Understand next state is chosen
5) Understand how an action is determined
6) Look at the slides 21-26 of lecture Week 9, how Q-values are calculated
7) Change time-steps and episodes in the code and understand their impact on the
game
'''
#https://github.com/MJeremy2017/RL/blob/master/GridWorld/gridWorld.py


#("DETERMINISTIC, STATE-VALUE LEARNING")
# Aspect             | Description
# Movement           | Deterministic (fixed moves: if you press "up", you move up).
# Learning           | State-value learning (learns the value of being at a position).
# Rewards            | Reward is assigned based on reaching a state.
# Policy             | Tries to choose the action that leads to a state with highest value.
# State-Action Info  | Only states are valued, actions are not separately learned.
# Update Formula     | Temporal Difference:
# # reward = previous reward + learning_rate × (next reward - previous reward) V(s)=V(s)+lr×(reward−V(s))
# | Exploration | Random actions taken 30% of the time (exploration rate). | | Backpropagation | Updates the value of each visited state after game ends. |
import numpy as np

# global variables
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = True
# Defines board size (3 rows, 4 columns).
# Winning position is (0,3).
# Losing position is (1,3).
# Starting position is (2,0).
# DETERMINISTIC movement: means movement is predicta

class State: #
    def __init__(self, state=START):#Constructor for State class. It initializes the board.
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS]) #Board filled with zeros initially (empty positions).
        self.board[1, 1] = -1 #The position (1,1) is a block/wall (cannot pass through, marked -1).
        self.state = state #              ]
        self.isEnd = False #              ]Sets the current position, marks game not ended, and movement deterministic.
        self.determine = DETERMINISTIC #  ]

    #And as a grid game, it needs a State to justify each state(position) of our agent, giving reward according to its state.
    def giveReward(self):
        # Returns reward based on current state:
        # +1 if Win,
        # -1 if Lose,
        # 0 otherwise.
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def isEndFunc(self): #Checks if the game has ended (win or lose).
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    #When our agent takes an action, the State should have a function to accept an action and return a legal position of next state.
    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        # Calculates next position based on action:
        # up: move up (row -1),
        # down: move down (row +1),
        # left: move left (column -1),
        # right: move right (column +1).
        # Also ensures:
        # Can't go outside the board,
        # Can't go into block (1,1).
        # If invalid move, stay in same place
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS -1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS -1)):
                    if nxtState != (1, 1):
                        return nxtState
            return self.state

    def showBoard(self):
        # Displays the board:
        # * = Current agent's position,
        # z = Block/wall,
        # 0 = Empty.
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


# Agent of player
# This is the artificial intelligence part, as our agent should be able to
#  learn from the process and thinks like a human. The key of the
#  magic is value iteration.
class Agent:
    # Constructor for Agent:
    # List of all moves done: self.states
    # All possible actions: self.actions
    # Current state: self.State
    # Learning rate self.lr = 0.2
    # Exploration rate self.exp_rate = 0.3 (30% random moves)
    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3

        # initial state reward
        self.state_values = {} #A dictionary storing values (scores) for all positions initialized to 0.
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0

    #  Exploration & Exploitation
    #  There is one last thing we need to talk about. Once our agent finds a
    #  path to get reward +1, should it sticks to it and forever follows that
    #  path (exploitation) or should it gives other path a chance(exploration)
    #  and expects a shorter path? In actual, we will balance exploration
    #  and exploitation in order to avoid our agent stuck in local optimal.
    #  Here our agent will choose action based on certain exploration_rate
    def chooseAction(self):
        # choose action with most expected value
        # Chooses action:
        # Exploration (random action with 30% probability).
        # Exploitation (choose the best action by looking at future state values).
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action): #Applies the action and returns a new State.
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self): #Resets the agent after each game.
        self.states = []
        self.State = State()

    # What our agent will finally learn is a policy, and a policy is a mapping
    #  from state to action, simply instructs what the agent should do at
    #  each state. In our case, instead of learning a mapping from state to
    #  action, we will leverage value iteration to firstly learn a mapping of
    #  state to value(which is the estimated reward) and based on the
    #  estimation, at each state, our agent will choose the best action that
    #  gives the highest estimated reward.
    #  There is not going to be any cranky, head-scratching math involved,
    #  as the core of value iteration is amazingly concise.
    # V(St)<-V(St) + a[V(St+1) - V(St)]

    #  At first, our gent knows nothing about the grid world(environment),
    #  so it would simply initialises all reward as 0. Then, it starts to
    #  explore the world by randomly walking around, surely it will endure
    #  lots of failure at the beginning, but that is totally fine. Once it
    #  reaches end of the game, either reward +1 or reward -1, the whole
    #  game reset and the reward propagates in a backward fashion and
    #  eventually the estimated value of all states along the way will be
    #  updated based on the formula above.

    #  Let’s take a closer look at the formula. The V(St) on the left is the
    #  updated value of that state, and the right one is the current non
    # updated value and α is learning rate. The formula is simply saying
    #  that the updated value of a state equals to the current value plus a
    #  temporal di!erence, which is what the agent learned from this
    #  iteration of game playing minus the previous estimate. For example,
    #  let’s say there are 2 states, S1 and S2 , both of which has an
    #  estimated value 0, and at this round of playing, our agent moves
    #  from S1 to S2 and gets reward 1, then the new estimate of S1 = S1
    #  + α(S2 - S1) , which is 0 + 0.1(1-0) = 0 (assume α is 0.1 and reward at
    #  S1 is 0)

    def play(self, rounds=10):
        # Main training loop:
        # Plays the game for given number of rounds.
        # After reaching end, backpropagates rewards through visited states.
        # How reward is updated:
        # reward = previous reward + learning_rate × (next reward - previous reward)
        # This is temporal difference learning.
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print("---------------------")

    def showValues(self): #Prints the learned values of all positions in a grid form.
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

# Main Code
# Creates an Agent.
# Makes it play 50 games.
# Displays final learned values of all the board positions.
if __name__ == "__main__":
    ag = Agent()
    ag.play(50)
    print(ag.showValues())



# Quick Summary:

# Component   | Purpose
# State class | Handles the board, position, rules, rewards.
# Agent class | Learns how to move optimally via exploration and exploitation.
# play        | Makes agent play, learn by updating value function.
# showValues  | Displays learned values (how good each position is).

