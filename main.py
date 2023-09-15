#!/usr/bin/env python3
import logging
import argparse

from policy.dog_policy import DogPolicy
from policy.npc_policy import NPCPolicy
from envs.AI_dog_env import AIDogEnv
from viewers.AI_dog_viewer import AIDogViewer

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_type", default = "linear")
    parser.add_argument("--reward_type", default = "simple")
    parser.add_argument("--times", default = 5e3, type = int)
    args = vars(parser.parse_args())
    return args

def main():
    args = parse_arg()
    env = AIDogEnv()
    env.reset()
    dog_policy = DogPolicy(args["net_type"], args["reward_type"])
    npc_policy = NPCPolicy()
    viewer = AIDogViewer(env, dog_policy, npc_policy)
    viewer.run(args["times"])

if __name__ == '__main__':
    main()
