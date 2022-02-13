import random
import numpy as np

class Environment:

    def __init__(self, direction, next, reward, initial_state_range=0, total_steps=10, initial_context=None, N_obs=None):
        self.direction = direction
        self.next = next
        self.reward = reward
        if N_obs is not None:
            self.N_obs = N_obs
        else:
            self.N_obs = len(next)
        self.N_act = len(direction)
        self.total_steps = total_steps
        self.initial_state_range = initial_state_range
        self.initial_context = initial_context
        self.reset()

    def reset(self):
        self.state = random.randint(0, self.initial_state_range)
        self.time = 0
        self.steps_left = self.total_steps
        if self.initial_context is not None:
            self.context = self.initial_context.copy()
        self.final_condition = False
        return self.state


    def get_directions(self):
        return list(self.next[self.state].keys())

    def is_done(self):
        return self.steps_left == 0 or self.final_condition

    def step(self, action):
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        self.time += 1
        if type(self.next) is not dict:
            # Salesman
            self.state, reward, self.context = self.next(self.state, self.context, action)
            if sum(self.context) == 0:
                self.final_condition = True
            return self.state, reward, self.is_done(), None
        else:
            # Others
            if self.direction[action] in self.get_directions():
                dir = self.next[self.state][self.direction[action]]
                if type(dir) is tuple:
                    self.state = np.random.choice(dir[0], p=dir[1])
                    try: 
                        reward = self.reward[self.state]
                    except:
                        reward = self.reward[action, self.state]
                else:
                    self.state = dir
                    reward = self.reward[self.state]
                return self.state, reward, self.is_done(), None
            else:
                return self.state, 0, self.is_done(), None

    @classmethod
    def volleyBall(cls, a, b, initial_state_range=0, action_prior = None):
        direction = [0, 1]
        next = {
            0: {0: ((0, 1), (1-a, a)), 1: ((0, 1), (1-b, b))},
            1: {}
        }
        if action_prior is not None:
            reward = {(0,0):0, (1,0):action_prior, (0,1): 1, (1,1): 1+action_prior}
        else:
            reward = {0:0, 1:1}
        return cls(direction, next, reward, initial_state_range=initial_state_range, total_steps=1)

    @classmethod
    def tp1(cls, initial_state_range=4):

        direction = {
            0: "E",
            1: "S",
            2: "W",
            3: "N"
        }

        next = {
            0: {"S": 5, "E": 1},
            1: {"W": 0, "E": 2},
            2: {"W": 1, "S": 6, "E": 3},
            3: {"W": 2, "E": 4},
            4: {"W": 3, "S": 7},
            5: {"N": 0},
            6: {"N": 2},
            7: {"N": 4}
        }

        reward = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: -10, 6: 10, 7: -10}

        return cls(direction, next, reward, initial_state_range = initial_state_range, total_steps = 10)

    @classmethod
    def bottleneck(cls, initial_state_range=0):

        direction = {
            0: "E",
            1: "S",
            2: "W",
            3: "N"
        }

        next = {
            0: {"S": 6, "E": 1},
            1: {"W": 0, "S": 7, "E": 2},
            2: {"W": 1, "S": 8},
            3: {"S": 9, "E": 4},
            4: {"W": 3, "S": 10, "E": 5},
            5: {"W": 4, "S": 11},
            6: {"N": 0, "E": 7, "S": 12},
            7: {"N": 1, "E": 8, "S": 13, "W": 6},
            8: {"N": 2, "E": 9, "S": 14, "W": 7},
            9: {"N": 3, "E": 10, "S": 15, "W": 8},
            10: {"N": 4, "E": 11, "S": 16, "W": 9},
            11: {"N": 5, "S": 17, "W": 10},
            12: {"N": 6, "E": 13},
            13: {"W": 12, "N": 7, "E": 14},
            14: {"W": 13, "N": 8},
            15: {"N": 9, "E": 16},
            16: {"W": 15, "N": 10, "E": 17},
            17: {"W": 16, "N": 11}
        }

        reward = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                  9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 1}

        return cls(direction, next, reward, initial_state_range=initial_state_range, total_steps=7)
    
    @classmethod
    def bottleneck2(cls, initial_state_range=0):

        direction = {
            0: "E",
            1: "S",
            2: "W",
            3: "N"
        }

        next = {
            0: {"S": 6, "E": 1},
            1: {"W": 0, "S": 7, "E": 2},
            2: {"W": 1, "S": 8},
            3: {"S": 9, "E": 4},
            4: {"W": 3, "S": 10, "E": 5},
            5: {"W": 4, "S": 11},
            6: {"N": 0, "E": 7, "S": 12},
            7: {"N": 1, "E": 8, "S": 13, "W": 6},
            8: {"N": 2, "E": 9, "S": 14, "W": 7},
            9: {"N": 3, "E": 10, "S": 15, "W": 8},
            10: {"N": 4, "E": 11, "S": 16, "W": 9},
            11: {"N": 5, "S": 17, "W": 10},
            12: {"N": 6, "E": 13},
            13: {"W": 12, "N": 7, "E": 14},
            14: {"W": 13, "N": 8},
            15: {"N": 9, "E": 16},
            16: {"W": 15, "N": 10, "E": 17},
            17: {"W": 16, "N": 11}
        }

        reward = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0,
                  9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 1}

        return cls(direction, next, reward, initial_state_range=initial_state_range, total_steps=7)

    @classmethod
    def square(cls, initial_state_range=0, side = 10, index_reward = None):
        direction = {
            0: "E",
            1: "S",
            2: "W",
            3: "N"
        }

        next = {}
        reward = {}
        for i in range(side):
            for j in range(side):
                state = i * side + j
                reward[state] = 0
                next[state] = {}
                if j < side - 1:
                    next[state]["E"] = state + 1
                else:
                    next[state]["E"] = state
                if j > 0:
                    next[state]["W"] = state - 1
                else:
                    next[state]["W"] = state
                if i < side - 1:
                    next[state]["S"] = state + side
                else:
                    next[state]["S"] = state
                if i > 0:
                    next[state]["N"] = state - side
                else:
                    next[state]["N"] = state

        if index_reward is None:
            reward[side*side-1] = 1
        else:
            reward[index_reward] = 1

        print(next)
        print(reward)

        return cls(direction, next, reward, initial_state_range=initial_state_range, total_steps=2*(side-1))
    
    @classmethod
    def salesman(cls, initial_state_range=0, total_steps=6):
        direction = {
            0:1,
            1:2,
            2:3,
            3:4,
            4:5,
            5:6
        }
        
        def next(state, context, action):
            past_state = state
            if state != (action+1):
                state = action + 1
                context[action] = 0
                # distance calculation
                if past_state == 0:
                    x_0 = 0
                    y_0 = 0
                else:
                    theta_0 = (past_state - 1) * np.pi / 3
                    x_0 = np.cos(theta_0)
                    y_0 = np.sin(theta_0)
                theta_1 = (state - 1) * np.pi / 3
                x_1 = np.cos(theta_1)
                y_1 = np.sin(theta_1)
                dist = np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)
                reward = -dist - 1
            else:
                reward = -1 # time malus
            return state, reward, context           
        
        reward = None
        
        initial_context = [1, 1, 1, 1, 1, 1]
        
        return cls(direction, next, reward, 
                   initial_state_range=initial_state_range, 
                   total_steps=total_steps, 
                   initial_context=initial_context,
                   N_obs=7)


