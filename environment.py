import random

class Environment:

    def __init__(self, direction, next, reward, initial_state_range=0, total_steps=10):
        self.direction = direction
        self.next = next
        self.reward = reward
        self.N_obs = len(next)
        self.N_act = len(direction)
        self.total_steps = total_steps
        self.initial_state_range = initial_state_range
        self.reset()

    def reset(self):
        self.state = random.randint(0, self.initial_state_range)
        self.time = 0
        self.steps_left = self.total_steps
        return self.state

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


    def get_directions(self):
        return list(self.next[self.state].keys())

    def is_done(self):
        return self.steps_left == 0

    def step(self, action):
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        self.time += 1
        if self.direction[action] in self.get_directions():
            self.state = self.next[self.state][self.direction[action]]
            return self.state, self.reward[self.state], self.is_done(), None
        else:
            return self.state, 0, self.is_done(), None


