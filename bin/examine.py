#!/usr/bin/env python3
import logging

# from mujoco_worldgen.util.envs import examine_env, load_env
# from mujoco_worldgen.util.path import worldgen_path
# from mujoco_worldgen.util.parse_arguments import parse_arguments

from dog_policy.dog_policy import DogPolicy
from dog_envs.viewer.base_viewer import BaseViewer

from examples import ai_dog

import argparse

logger = logging.getLogger(__name__)


# For more detailed on argv information, please have a look on
# docstring below.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model", default = "dog.pt")
    parser.add_argument("--load_model", default = "dog.pt")
    parser.add_argument("--env_name", default = "ai_dog")
    parser.add_argument("--use_cuda", default = True)
    args = vars(parser.parse_args())

    assert(args["env_name"] == "ai_dog" and "env not match")
    env = ai_dog.make_env()
    env.reset()
    dog_policy = DogPolicy(4, 2, args["load_model"], args["use_cuda"])
    viewer = BaseViewer(env, dog_policy)
    viewer.run(5e2)
    dog_policy.save(args["save_model"])

if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
