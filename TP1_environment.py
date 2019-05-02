import random

class Environment:

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

    def __init__(self):
        self.state = random.randint(0, 4)
        self.steps_left = 10
        self.N_obs = len(Environment.next)
        self.N_act = len(Environment.direction)

    def get_observation(self):
        return self.state

    def get_directions(self):
        return list(Environment.next[self.state].keys())

    def is_done(self):
        return self.steps_left == 0

    def act(self, action):
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        if Environment.direction[action] in self.get_directions():
            self.state = Environment.next[self.state][Environment.direction[action]]
            return self.state, Environment.reward[self.state], self.is_done()
        else:
            return self.state, 0, self.is_done()


