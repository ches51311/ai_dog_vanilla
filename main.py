#!/usr/bin/env python3
import logging
import argparse
import os
import site
site.addsitedir(os.path.join(os.path.dirname(__file__), "policy"))

from dog_vital_sign import DogVitalSign
from dog_reward import DogReward
from dog_policy import DogPolicy
from npc_policy import NPCPolicy
from envs.AI_dog_env import AIDogEnv
from viewers.AI_dog_viewer import AIDogViewer

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_type", default = "linear")
    parser.add_argument("--reward_type", default = "simple")
    parser.add_argument("--use_critic", default  = "False", action="store_true")
    parser.add_argument("--times", default = 5e3, type = int)
    args = vars(parser.parse_args())
    return args

def main():
    args = parse_arg()
    env = AIDogEnv()
    env.reset()
    dog_vital_sign = DogVitalSign(args["reward_type"])
    dog_reward = DogReward(dog_vital_sign, args["reward_type"], args["use_critic"])
    dog_policy = DogPolicy(dog_reward, args["net_type"])
    npc_policy = NPCPolicy()
    viewer = AIDogViewer(env, dog_vital_sign, dog_reward, dog_policy, npc_policy)
    viewer.run(args["times"])

if __name__ == '__main__':
    main()
