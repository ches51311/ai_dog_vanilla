from mujoco_worldgen.util.envs import EnvViewer
from mujoco_py import const, ignore_mujoco_warnings
import glfw
import matplotlib.pyplot as plt
import numpy as np

class AIDogViewer(EnvViewer):
    def __init__(self, env, dog_policy, npc_policy):
        super().__init__(env = env)
        self.dog_policy = dog_policy
        self.npc_policy = npc_policy
        self.seed[0] = 0
        self.user_mode = False
        self.user_action = {"move": np.zeros(3, np.float32),
                            "feed": False,
                            "touch": False}
        self.obs = {"man_site": self.env.sim.data.get_site_xpos("man"),
                    "man_action": {"move": np.zeros(3, np.float32),
                                   "feed": False,
                                   "touch": False},
                    "dog_site": self.env.sim.data.get_site_xpos("AI_dog"),
                    "dog_action": {"move": np.zeros(3, np.float32),
                                   "bark": 0,
                                   "shake": 0}}


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
        self.add_overlay(const.GRID_TOPRIGHT, "dog's death:", str(self.dog_policy.death()))
        self.add_overlay(const.GRID_TOPRIGHT, "dog's Hungry Point:", str(self.dog_policy.hp))
        self.add_overlay(const.GRID_TOPRIGHT, "dog's Mood Point:", str(self.dog_policy.mp))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "dog's move", str(self.obs["dog_action"]["move"]))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "dog's bark", str(self.obs["dog_action"]["bark"]))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "dog's shake", str(self.obs["dog_action"]["shake"]))
        # self.add_overlay(const.GRID_BOTTOMRIGHT, "man_site", str(self.obs["man_site"]))
        # self.add_overlay(const.GRID_BOTTOMRIGHT, "dog_site", str(self.obs["dog_site"]))
        # self.add_overlay(const.GRID_BOTTOMRIGHT, "man_action", str(self.obs["man_action"]))
        # self.add_overlay(const.GRID_BOTTOMRIGHT, "dog_action", str(self.obs["dog_action"]))
        self.render()

    def run(self, total_num_episodes = 5e2, reborn_cnt = 500):
        losses = []

        for episode in range(int(total_num_episodes)):
            print("Run episode:", episode)

            while not self.dog_policy.death():
                with ignore_mujoco_warnings():
                    self.dog_policy.my_turn(self.obs)
                    if self.user_mode:
                        self.obs["man_action"] = self.user_action
                    else:
                        self.npc_policy.my_turn(self.obs)
                    self.env.my_turn(self.obs)
                    self.show_info()
            self.obs["dog_action"]["move"] = np.zeros(3, np.float32)
            for i in range(reborn_cnt):
                with ignore_mujoco_warnings():
                    self.env.my_turn(self.obs)
                    self.show_info()
            self.dog_policy.reborn()
            self.env_reset()

            # self.dog_policy.update()
            # print("actual_reward_sum:", self.dog_policy.actual_reward_sum)
            # print("loss:", self.dog_policy.loss)
            # losses.append(self.dog_policy.loss)
        # plt.plot(losses)
        # plt.show()
