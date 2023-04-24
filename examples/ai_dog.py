import numpy as np
from mujoco_worldgen import WorldParams, WorldBuilder, Floor, ObjFromXML
from dog_envs.base import Base


def get_reward(sim):
    object_xpos = sim.data.get_site_xpos("NPC")
    # target_xpos = sim.data.get_site_xpos("target")
    target_xpos = sim.data.get_site_xpos("AI_dog")
    ctrl = np.sum(np.square(sim.data.ctrl))
    return -np.sum(np.square(object_xpos - target_xpos)) - 1e-3 * ctrl


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
    return Base(get_sim=get_sim, get_reward=get_reward, horizon=30)
