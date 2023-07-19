from mujoco_worldgen import Env
from mujoco_worldgen import WorldParams, WorldBuilder, Floor, ObjFromXML
import numpy as np

class AIDogEnv(Env):
    def __init__(self,
                 horizon=1000):
        super().__init__(get_sim = self._get_sim,
                         horizon = horizon)

    def _get_sim(self, seed):
        world_params = WorldParams(size=(4., 4., 2.5))
        builder = WorldBuilder(world_params, seed)
        floor = Floor()
        builder.append(floor)
        man = ObjFromXML("particle_hinge")
        dog = ObjFromXML("particle_hinge_dog")
        floor.append(man)
        floor.append(dog)
        man.mark("man")
        dog.mark("AI_dog")
        return builder.get_sim()

    def my_turn(self, obs):
        action = np.concatenate([obs["man_action"]["move"], 
                                 obs["dog_action"]["move"]]).astype(np.float32)
        _, _, _, _ = self.step(action)
        obs["man_site"] = self.sim.data.get_site_xpos("man")
        obs["dog_site"] = self.sim.data.get_site_xpos("AI_dog")


