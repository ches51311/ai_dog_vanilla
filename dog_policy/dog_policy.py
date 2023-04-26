from copy import deepcopy
import numpy as np

class DogPolicy:
    def __init__(self, ob_space, ac_space):
        self.ob_space = ob_space
        self.ac_space = ac_space
    def act(self, observation):
        obs = deepcopy(observation)
        cal = (obs['npc_site'] - obs['dog_site'])
        if sum(abs(cal)) < 0.5:
            action = [0, 0]
        else:
            cal = cal*0.05/max(abs(cal))
            action = [cal[0], cal[1]]
        return np.array(action, np.float32)