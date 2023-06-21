from mujoco_worldgen.util.envs import EnvViewer
from mujoco_py import const, ignore_mujoco_warnings
import glfw
import matplotlib.pyplot as plt
import numpy as np

class BaseViewer(EnvViewer):
    def __init__(self, env, policy):
        super().__init__(env = env)
        self.dog_policy = policy
        self.seed[0] = 0
        self.num_npc = 1
        self.num_dog = 1
        self.npc_action = np.zeros(3 * self.num_npc, np.float32)
        self.dog_action = np.zeros(3 * self.num_dog, np.float32)

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Cancel Advances simulation by one step called from super
        if self._advance_by_one_step == True:
            self._advance_by_one_step = False
            self._paused = False

        if key == glfw.KEY_LEFT:
            self.npc_action[0] = -0.2
            self.npc_action[1] = 0
        elif key == glfw.KEY_RIGHT:
            self.npc_action[0] = 0.2
            self.npc_action[1] = 0
        elif key == glfw.KEY_UP:
            self.npc_action[1] = 0.2
            self.npc_action[0] = 0
        elif key == glfw.KEY_DOWN:
            self.npc_action[1] = -0.2
            self.npc_action[0] = 0
        elif key == glfw.KEY_J: # rotate
            self.npc_action[2] = 0.3
        elif key == glfw.KEY_K:
            self.npc_action[2] = -0.3
        elif key == glfw.KEY_SPACE:
            self.npc_action = np.zeros(3 * self.num_npc, np.float32)

    def get_dog_obs(self):
        npc_site = self.env.sim.data.get_site_xpos("NPC")
        dog_site = self.env.sim.data.get_site_xpos("AI_dog")
        dog_obs = [npc_site[0], npc_site[1], dog_site[0], dog_site[1]]
        return dog_obs
    def set_seed(self):
        self.seed[0] += 1
        self.seed[0] %= 20
        self.env.seed(self.seed)

    def run(self, total_num_episodes = 5e2):
        dog_obs = self.get_dog_obs()
        losses = []
        for episode in range(int(total_num_episodes)):
            self.set_seed()
            self.env_reset()
            self.npc_action = np.zeros(3 * self.num_npc, np.float32)
            self.dog_action = np.zeros(3 * self.num_dog, np.float32)
            print("Run episode:", episode)

            done = False
            while not done:
                with ignore_mujoco_warnings():
                    dog_obs = self.get_dog_obs()
                    self.dog_action[:2] = self.dog_policy.action(dog_obs)
                    action = np.concatenate([self.npc_action, self.dog_action])
                    obs, reward, done, info = self.env.step(action)
                    self.dog_policy.add_reward(reward)
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_BOTTOMRIGHT, "NPC Action", str(self.npc_action))
                self.add_overlay(const.GRID_BOTTOMRIGHT, "dog Action", str(self.dog_action))
                self.render()
            self.dog_policy.update()
            print("actual_reward_sum:", self.dog_policy.actual_reward_sum)
            print("loss:", self.dog_policy.loss)
            losses.append(self.dog_policy.loss)
        plt.plot(losses)
        plt.show()
