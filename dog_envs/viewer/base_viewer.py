from mujoco_worldgen.util.envs import EnvViewer
from mujoco_py import const, ignore_mujoco_warnings
import glfw

class BaseViewer(EnvViewer):
    def __init__(self, env, policy):
        super().__init__(env = env)
        self.dog_policy = policy

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Cancel Advances simulation by one step called from super
        if self._advance_by_one_step == True:
            self._advance_by_one_step = False
            self._paused = False

        if key == glfw.KEY_LEFT:
            self.action[0] = -0.2
            self.action[1] = 0
        elif key == glfw.KEY_RIGHT:
            self.action[0] = 0.2
            self.action[1] = 0
        elif key == glfw.KEY_UP:
            self.action[1] = 0.2
            self.action[0] = 0
        elif key == glfw.KEY_DOWN:
            self.action[1] = -0.2
            self.action[0] = 0
        elif key == glfw.KEY_J: # rotate
            self.action[2] = 0.3
        elif key == glfw.KEY_K:
            self.action[2] = -0.3
        elif key == glfw.KEY_SPACE:
            self.action = self.zero_action(self.env.action_space)

    def run(self, once=False):
        total_num_episodes = int(5e3)
        obs, reward, done, info = self.env.step(self.action)
        for episode in range(total_num_episodes):
            self.seed[0] += 1
            self.env.seed(self.seed)
            self.env_reset()
            self.action = self.zero_action(self.env.action_space)
            print("Run episode:", episode)

            done = False
            while not done:
                with ignore_mujoco_warnings():
                    self.action[3:5] = self.dog_policy.act(obs)
                    obs, reward, done, info = self.env.step(self.action)
                    self.dog_policy.add_reward(reward)
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Apply action", "A (-0.05) / Z (+0.05)")
                self.add_overlay(const.GRID_TOPRIGHT, "on action index %d out %d" % (self.action_mod_index, self.num_action), "J / K")
                self.add_overlay(const.GRID_BOTTOMRIGHT, "Reset took", "%.2f sec." % (sum(self.elapsed) / len(self.elapsed)))
                self.add_overlay(const.GRID_BOTTOMRIGHT, "Action", str(self.action))
                self.render()
                if once:
                    return
            self.dog_policy.update_dog()
