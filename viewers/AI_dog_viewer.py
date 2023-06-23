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
        self.user_mode = True
        self.user_action = np.zeros(3, np.float32)

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Cancel Advances simulation by one step called from super
        if self._advance_by_one_step == True:
            self._advance_by_one_step = False
            self._paused = False
        if action != glfw.RELEASE:
            return
        if key == glfw.KEY_LEFT:
            self.user_action[0] = -0.2
            self.user_action[1] = 0
        elif key == glfw.KEY_RIGHT:
            self.user_action[0] = 0.2
            self.user_action[1] = 0
        elif key == glfw.KEY_UP:
            self.user_action[1] = 0.2
            self.user_action[0] = 0
        elif key == glfw.KEY_DOWN:
            self.user_action[1] = -0.2
            self.user_action[0] = 0
        elif key == glfw.KEY_J: # rotate
            self.user_action[2] = 0.3
        elif key == glfw.KEY_K:
            self.user_action[2] = -0.3
        elif key == glfw.KEY_U:
            self.user_action = np.zeros(3, np.float32)
            self.user_mode = not self.user_mode

    def set_seed(self):
        self.seed[0] += 1
        self.seed[0] %= 20
        self.env.seed(self.seed)

    def get_dog_obs(self):
        npc_site = self.env.sim.data.get_site_xpos("NPC")
        dog_site = self.env.sim.data.get_site_xpos("AI_dog")
        dog_obs = [npc_site[0], npc_site[1], dog_site[0], dog_site[1]]
        return dog_obs

    def get_npc_obs(self):
        pass

    # self.dog_action, self.npc_action
    # cur_iter => for dog to update net
    def update_dog_state(self):
        pass

    def update_npc_state(self):
        if self.user_mode:
            return
        pass

    def update_env(self):
        if self.user_mode:
            self.man_action = self.user_action
        else:
            self.man_action = self.npc_action
        action = np.concatenate([self.man_action, self.dog_action]).astype(np.float32)
        # done if over horizion
        _, _, done, _ = self.env.step(action)
        self.done = done or self.done
    
    def show_info(self):
        self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
        self.add_overlay(const.GRID_BOTTOMRIGHT, "man Action", str(self.man_action))
        self.add_overlay(const.GRID_BOTTOMRIGHT, "dog Action", str(self.dog_action))
        self.render()

    def run(self, total_num_episodes = 5e2):
        dog_obs = self.get_dog_obs()
        losses = []

        for episode in range(int(total_num_episodes)):
            self.set_seed()
            self.env_reset()
            print("Run episode:", episode)

            self.done = False
            while not self.done:
                with ignore_mujoco_warnings():
                    # 1. dog decide what to do
                    dog_obs = self.get_dog_obs()
                    self.dog_action = self.dog_policy.action(dog_obs)
                    # 2. npc decide what to do
                    npc_obs = self.get_npc_obs()
                    self.npc_action = self.npc_policy.action(npc_obs)
                    # 3. update env
                    self.update_dog_state()
                    self.update_npc_state()
                    self.update_env()
                    # 4. get reward
                    self.dog_policy.set_reward(self.env.sim)
                    self.done = self.done or self.dog_policy.get_diverged()
                    # show info
                    self.show_info()

            self.dog_policy.update()
            print("actual_reward_sum:", self.dog_policy.actual_reward_sum)
            print("loss:", self.dog_policy.loss)
            losses.append(self.dog_policy.loss)
        plt.plot(losses)
        plt.show()
