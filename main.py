#!/usr/bin/env python3
import logging
import argparse

from policy.dog_policy import DogPolicy
from policy.npc_policy import NPCPolicy
from envs.AI_dog_env import AIDogEnv
from viewers.AI_dog_viewer import AIDogViewer

logger = logging.getLogger(__name__)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model", default = "dog.pt")
    parser.add_argument("--load_model", default = "dog.pt")
    parser.add_argument("--rewards", default = "rewards.json")
    parser.add_argument("--no_cuda", action = "store_true")
    parser.add_argument("--times", default = 5e2, type = int)
    args = vars(parser.parse_args())
    return args

def main():
    args = parse_arg()
    env = AIDogEnv()
    env.reset()
    dog_policy = DogPolicy(args["load_model"], args["no_cuda"], True, args["save_model"])
    npc_policy = NPCPolicy()
    viewer = AIDogViewer(env, dog_policy, npc_policy)
    viewer.run(args["times"])
    dog_policy.save(args["save_model"])
    dog_policy.save_reward_history(args["rewards"])

if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
