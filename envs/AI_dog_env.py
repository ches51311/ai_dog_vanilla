from mujoco_worldgen import Env
from mujoco_worldgen import WorldParams, WorldBuilder, Floor, ObjFromXML
import numpy as np

class AIDogEnv(Env):
    def __init__(self, horizon=1000):
        super().__init__(get_sim = self._get_sim, horizon = horizon)

    def _get_sim(self, seed):
        floor_length = 4.
        floor_width = 4.
        world_params = WorldParams(size=(floor_length, floor_width, 2.5))
        builder = WorldBuilder(world_params, seed)
        floor = Floor()
        builder.append(floor)
        breeder = ObjFromXML("particle_hinge")
        dog = ObjFromXML("particle_hinge_dog")
        floor.append(breeder)
        floor.append(dog)
        breeder.mark("breeder")
        dog.mark("AI_dog")
        return builder.get_sim()

    def my_turn(self, obs):
        action = np.concatenate([obs["breeder_action"]["move"], 
                                 obs["dog_action"]["move"]]).astype(np.float32)
        _, _, _, _ = self.step(action)
        obs["breeder_site"] = self.sim.data.get_site_xpos("breeder")
        obs["dog_site"] = self.sim.data.get_site_xpos("AI_dog")


