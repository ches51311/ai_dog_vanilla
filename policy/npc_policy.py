import numpy as np

class NPCPolicy:
    def __init__(self):
        pass
    def action(self, npc_obs):
        return np.zeros(3, np.float32)
