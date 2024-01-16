from mujoco_worldgen.util.envs import EnvViewer
from mujoco_py import const, ignore_mujoco_warnings
import glfw
import matplotlib.pyplot as plt
import numpy as np
import time
import copy

class AIDogViewer(EnvViewer):
    def __init__(self, env, dog_env, dog_reward, dog_policy, npc_policy):
        super().__init__(env = env)
        self.dog_env = dog_env
        self.dog_reward = dog_reward
        self.dog_policy = dog_policy
        self.npc_policy = npc_policy
        self.seed[0] = 0
        self.user_mode = False
        self.user_action = {"move": np.zeros(3, np.float32),
                            "feed": False,
                            "touch": False}
        self.obs = {"breeder_site": self.env.sim.data.get_site_xpos("breeder"),
                    "breeder_action": {"move": np.zeros(3, np.float32),
                                   "feed": False,
                                   "touch": False},
                    "dog_site": self.env.sim.data.get_site_xpos("AI_dog"),
                    "dog_action": {"move": np.zeros(3, np.float32),
                                   "bark": 0,
                                   "shake": 0,
                                   "prob": None}}
        self.prev_obs = copy.copy(self.obs)

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Cancel Advances simulation by one step called from super
        if self._advance_by_one_step == True:
            self._advance_by_one_step = False
            self._paused = False
        if action != glfw.RELEASE:
            return
        if key == glfw.KEY_LEFT:
            self.user_action["move"][0] = -0.2
            self.user_action["move"][1] = 0
        elif key == glfw.KEY_RIGHT:
            self.user_action["move"][0] = 0.2
            self.user_action["move"][1] = 0
        elif key == glfw.KEY_UP:
            self.user_action["move"][1] = 0.2
            self.user_action["move"][0] = 0
        elif key == glfw.KEY_DOWN:
            self.user_action["move"][1] = -0.2
            self.user_action["move"][0] = 0
        elif key == glfw.KEY_J: # rotate
            self.user_action["move"][2] = 0.3
        elif key == glfw.KEY_K:
            self.user_action["move"][2] = -0.3
        elif key == glfw.KEY_U:
            self.user_action["move"] = np.zeros(3, np.float32)
            self.user_mode = not self.user_mode

    def show_info(self):
        self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
        self.add_overlay(const.GRID_TOPRIGHT, "dog's state:", str(self.dog_env.state))
        self.add_overlay(const.GRID_TOPRIGHT, "dog's Hungry Point:", str(self.dog_env.hp))
        self.add_overlay(const.GRID_TOPRIGHT, "dog's Mood Point:", str(self.dog_env.mp))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "dog's move", str(self.obs["dog_action"]["move"]))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "dog's bark", str(self.obs["dog_action"]["bark"]))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "dog's shake", str(self.obs["dog_action"]["shake"]))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "breeder_site", str(self.obs["breeder_site"]))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "dog_site", str(self.obs["dog_site"]))
        # self.add_overlay(const.GRID_BOTTOMRIGHT, "breeder_action", str(self.obs["breeder_action"]))
        # self.add_overlay(const.GRID_BOTTOMRIGHT, "dog_action", str(self.obs["dog_action"]))
        self.render()

    def run(self, times = 5e2, max_reset = 10000):
        reincarnation = 0
        cnt = 0
        while self.dog_policy.dog_updated_cnt < times:
            cnt += 1
            with ignore_mujoco_warnings():
                if cnt > max_reset:
                    self.env_reset()
                    cnt = 0
                if self.dog_env.state == "just_reborn":
                    self.env_reset()
                    cnt = 0
                    reincarnation += 1
                    print("New life:", reincarnation)

                # Action: both Breeder & Dog do the action
                self.dog_policy.my_turn(self.obs)
                if self.user_mode:
                    self.obs["breeder_action"] = self.user_action
                else:
                    self.npc_policy.my_turn(self.obs)
                # State: both update the sandbox environment and dog's physiological function
                self.env.my_turn(self.obs)
                self.dog_env.my_turn(self.obs)
                # Reward: According to the current state, score dog'S reward
                self.dog_reward.my_turn(self.obs)
                if self.dog_env.state == "just_death":
                    print("Survive:", self.dog_env.life)

                self.show_info()

        self.dog_reward.plot_rewards()