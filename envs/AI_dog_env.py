from mujoco_worldgen import Env
from mujoco_worldgen.util.sim_funcs import (
    empty_get_info,
    flatten_get_obs,
    ctrl_set_action,
    )
from mujoco_worldgen import WorldParams, WorldBuilder, Floor, ObjFromXML

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
        npc = ObjFromXML("particle_hinge")
        dog = ObjFromXML("particle_hinge_dog")
        floor.append(npc)
        floor.append(dog)
        npc.mark("NPC")
        dog.mark("AI_dog")
        return builder.get_sim()



