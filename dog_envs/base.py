from mujoco_worldgen import Env
from mujoco_worldgen.util.sim_funcs import (
    empty_get_info,
    flatten_get_obs,
    false_get_diverged,
    ctrl_set_action,
    zero_get_reward,
    )
from copy import deepcopy

class Base(Env):
    def __init__(self,
                 get_sim,
                 get_obs=flatten_get_obs,
                 get_reward=zero_get_reward,
                 get_info=empty_get_info,
                 get_diverged=false_get_diverged,
                 set_action=ctrl_set_action,
                 action_space=None,
                 horizon=100,
                 start_seed=None,
                 deterministic_mode=False):
        super().__init__(get_sim = get_sim,
                         get_reward = get_reward,
                         get_diverged = get_diverged,
                         horizon = horizon)