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
        self.env_initialize()

    def env_initialize(self):
        self.state = random.randint(0, self.initial_state_range)
        self.steps_left = self.total_steps

    @classmethod
    def tp1(cls):

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

        initial_state_range = 4

        return cls(direction, next, reward, initial_state_range = initial_state_range, total_steps = 10)

    def get_observation(self):
        return self.state

    def get_directions(self):
        return list(self.next[self.state].keys())

    def is_done(self):
        return self.steps_left == 0

    def act(self, action):
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        if self.direction[action] in self.get_directions():
            self.state = self.next[self.state][self.direction[action]]
            return self.state, self.reward[self.state], self.is_done()
        else:
            return self.state, 0, self.is_done()


