import numpy as np
from mujoco_worldgen import WorldParams, WorldBuilder, Floor, ObjFromXML
from dog_envs.base import Base


def get_reward(sim):
    npc_site = sim.data.get_site_xpos("NPC")
    dog_site = sim.data.get_site_xpos("AI_dog")
    cal = sum(pow((npc_site - dog_site)[:2], 2))
    if cal < 3:
        return 1./(cal+0.1)
    else:
        return -(cal-3)*15.

def get_diverged(sim):
    npc_site = sim.data.get_site_xpos("NPC")
    dog_site = sim.data.get_site_xpos("AI_dog")
    cal = sum(pow((npc_site - dog_site)[:2], 2))
    if cal < 0.1:
        return True, 100.
    if cal > 7:
        return True, -100.
    return False, 0.




def get_sim(seed):
    world_params = WorldParams(size=(4., 4., 2.5))
    builder = WorldBuilder(world_params, seed)
    floor = Floor()
    builder.append(floor)
    obj = ObjFromXML("particle_hinge")
    obj2 = ObjFromXML("particle_hinge_dog")
    floor.append(obj)
    floor.append(obj2)
    obj.mark("NPC")
    obj2.mark("AI_dog")
    # floor.mark("target", (.5, .5, 0.05))
    return builder.get_sim()


def make_env():
    return Base(get_sim=get_sim, get_reward=get_reward, get_diverged = get_diverged, horizon=1000)
