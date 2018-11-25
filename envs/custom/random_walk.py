""" https://github.com/nczempin/gym-random-walk/blob/master/gym_random_walk/envs/random_walk_env.py  """

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym.envs.toy_text import discrete
import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

from string import ascii_uppercase
metadata = {'render.modes': ['human', 'ansi']}
LEFT, RIGHT = 0, 1

class RandomWalkEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends.

    Observations:
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations.

    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger

    Rewards:
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.


    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, n_states=7):

        self.nS = nS = n_states
        isd = np.zeros(nS)
        self.nA = nA = 2
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        for si in range(nS):
            for a in range(nA):

                if si > 0 and si < nS-1:
                    sf = si-1 if a == 0 else si + 1
                    sf = min(sf, nS-1) # never move than 6
                    sf = max(sf, 0) # never less than 0
                else:
                    sf = si

                r = 1. if si + 1 == sf and sf == nS-1 else 0.
                done = True if sf == 0 or sf == nS-1 else False

                pSucc = 0.500
                pFail = 1. - pSucc
                P[si][a].append((pSucc, sf, r, done))

        # Calculate initial state distribution
        # We always start in state (3)
        isd = np.zeros(nS)
        isd[3] = 1.0 # always start at same initial state

        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)



    def encode(self):
        raise NotImplementedError("Encode observations into a state object")


    def matrices(self):
        P = np.zeros((self.nA, self.nS, self.nS))
        R = np.zeros((self.nA, self.nS, self.nS))

        for si in range(self.nS):
            for a in range(self.nA):
                sf = self.P[si][a][0][1]
                pSucc = self.P[si][a][0][0]
                P[a, si, sf] += pSucc
                R[a, si, sf] += self.P[si][a][0][2]

                for other_a in range(self.nA):
                    if other_a != a:
                        other_sf = self.P[si][other_a][0][1]
                        P[a, si, other_sf] += (1. - pSucc)
                        R[a, si, other_sf] += self.P[si][other_a][0][2]


        return P, R



    # def step(self, a):
    #     a = np.random.choice([0,1], 1, p=[0.5,0.5])[0]
    #     return super(RandomWalkEnv, self).step(a)


    # def step(self, a):
    #
    #     return
    #
    #     # ----------------------------------------------------------------
    #     # Make the problem stochastic by successfully moving with p=0.7
    #     #   Pickup and drop-off actions remain deterministic
    #     # ----------------------------------------------------------------
    #     if self.stochastic:
    #         A = list(range(4))
    #         if a <= 3:
    #             a = np.random.choice(A, 1, p=[0.7 if a == ai else 0.1 for ai in A])[0]
    #             assert a >= 0 and a <= 3
    #
    #     return super(TaxiEnv, self).step(a)



    def render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = np.asarray([ascii_uppercase[:self.shape[1]]], dtype='c').tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        color = 'red' if self.s == 0 else 'green' if self.s == self.nS - 1 else 'yellow'
        desc[0][self.s] = utils.colorize(desc[0][self.s], color, highlight=True)
        outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile



    # def reset(self):
    #     # print("#self.size:",self.nS)
    #     self.state =  self.start_state_index
    #     # print("starting: ", self.state)