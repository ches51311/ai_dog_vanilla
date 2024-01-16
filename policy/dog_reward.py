import os
import math
import json
import torch
import matplotlib.pyplot as plt
from dog_network import *

torch.manual_seed(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DogReward:
    def __init__(self, dog_vital_sign,
                 reward_type = "simple",
                 use_critic = False,
                 dog_updated_freq = 100,
                 critic_updated_freq = 100,
                 critic_saved_freq = 50000,
                 log_reward_mean_N = 100,
                 log_saved_freq = 50000,
                 reward_max_len = 200,
                 saved_dir = os.path.join(os.getcwd(), "models")) -> None:
        self.dog_vital_sign = dog_vital_sign
        self.reward_type = reward_type
        self.use_critic = use_critic
        self.dog_updated_freq = dog_updated_freq
        self.critic_updated_freq = critic_updated_freq
        self.critic_saved_freq = critic_saved_freq
        self.log_reward_mean_N = log_reward_mean_N
        self.log_saved_freq = log_saved_freq
        self.reward_max_len = reward_max_len
        self.saved_dir = saved_dir

        self.my_turn_cnt = 0
        self.rewards_mean_cnt = 0
        self.log_filename = os.path.join(self.saved_dir, f"log_{self.reward_type}")

        if self.use_critic:
            self._set_model()

        self.prev_distance = None
        self.prev_obs = None
        self.raw_rewards = []
        self.rewards = []
        self.log_raw_rewards_means = []
        self.log_rewards_means = []

    def _set_model(self):
        # critic net is to estimated the rewards sum per dog_updated_freq
        self.critic_model_path = os.path.join(self.saved_dir, f"critic_{self.reward_type}.pth")
        if os.path.exists(self.critic_model_path):
            self.critic_net = torch.load(self.critic_model_path)
        else:
            self.critic_net = Critic().to(device)
        self.learning_rate = 1e-4
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=self.learning_rate) # dog_reward
        self.critic_loss_fn = torch.nn.SmoothL1Loss()
        self.critic_sum_rewards = []

        self.critic_net.train()

    def my_turn(self, obs):
        if self.dog_vital_sign.state in ["death", "just_reborn"]:
            return

        self.my_turn_cnt += 1

        if self.use_critic:
            if self.my_turn_cnt % self.critic_updated_freq == 1 and self.my_turn_cnt > 1:
                self._update_critic_weight()
            if self.my_turn_cnt % self.critic_saved_freq == 1 and self.my_turn_cnt > 1:
                self.save_critic(self.critic_model_path)

        self._set_raw_reward(obs)
        self._set_reward(obs)

        if self.my_turn_cnt % self.log_reward_mean_N == 1 and self.my_turn_cnt > 1:
            self._set_rewards_mean()
        if self.my_turn_cnt % self.log_saved_freq == 1 and self.my_turn_cnt > 1:
            self.save_log(self.log_filename)

        self.raw_rewards = self.raw_rewards[-self.reward_max_len:]
        self.rewards = self.rewards[-self.reward_max_len:]
        if self.use_critic:
            self.critic_sum_rewards = self.critic_sum_rewards[-self.reward_max_len:]

    def _set_raw_reward(self, obs):
        if self.dog_vital_sign.state == "just_death":
            raw_reward = -44444.+min(20000, self.dog_vital_sign.life)
        elif self.reward_type == "sai":
            raw_reward = (sum(abs(self.dog_action["move"] - self.sai_action["move"])) +
                      sum(abs(self.dog_action["bark"] - self.sai_action["bark"])) +
                      sum(abs(self.dog_action["shake"] - self.sai_action["shake"]))) * -1
        elif self.reward_type == "simple":
            distance = math.dist(obs["breeder_site"][:2], obs["dog_site"][:2])
            if self.prev_distance is None:
                self.prev_distance = distance
            raw_reward = (self.prev_distance - distance) * 1000
            self.prev_distance = distance
        elif self.reward_type == "HP_MP":
            raw_reward = (abs(self.dog_vital_sign.hp - 80) + abs(self.dog_vital_sign.mp - 80)) * -1 + min(5000, self.dog_vital_sign.life * 0.03)
        else:
            assert(0 and "reward_type illegal")
        self.raw_rewards.append(raw_reward)

    def _set_reward(self, obs):
        raw_reward = self.raw_rewards[-1]
        if self.use_critic == False:
            reward = raw_reward
        else:
            states = self.critic_net.concat_states(obs["dog_site"], obs["breeder_site"], \
                                            self.dog_vital_sign.hp, self.dog_vital_sign.mp, self.dog_vital_sign.life)
            critic_sum_reward = self.critic_net(states.to(device))
            if len(self.critic_sum_rewards) != 0:
                prev_critic_sum_reward = self.critic_sum_rewards[-1]
                factor = ((self.dog_updated_freq-1) / self.dog_updated_freq)
                # prev_critic_sum_reward: previous step's estimated sum
                # raw_reward + critic_sum_reward * factor: this step's estimated sum, replace one step's reward as raw_reward
                # so minus one step (i.e., mul factor)
                reward = raw_reward + critic_sum_reward * factor - prev_critic_sum_reward
            else:
                reward = raw_reward
            self.critic_sum_rewards.append(critic_sum_reward)


        self.rewards.append(reward)

    def _set_rewards_mean(self):
        raw_rewards_mean = torch.tensor(self.raw_rewards[-self.log_reward_mean_N:]).float().mean()
        rewards_mean = torch.tensor(self.rewards[-self.log_reward_mean_N:]).float().mean()
        print(self.rewards_mean_cnt , ":mean(raw_rewards):", raw_rewards_mean)
        print(self.rewards_mean_cnt , ":mean(rewards):", rewards_mean)
        self.log_raw_rewards_means.append(raw_rewards_mean)
        self.log_rewards_means.append(rewards_mean)

        self.rewards_mean_cnt += 1

    def _update_critic_weight(self):
        min_require_len = self.dog_updated_freq + self.critic_updated_freq
        if len(self.raw_rewards) < min_require_len:
            return
        critic_sum_rewards_for_update = []
        raw_sum_rewards_for_update = []
        critic_sum_start_site = len(self.critic_sum_rewards) - (self.dog_updated_freq + self.critic_updated_freq)
        raw_start_site = len(self.raw_rewards) - (self.dog_updated_freq + self.critic_updated_freq)
        # breakpoint()
        for i in range(self.critic_updated_freq):
            critic_sum_reward = self.critic_sum_rewards[critic_sum_start_site + i]
            raw_sum_reward = sum(self.raw_rewards[raw_start_site+i:raw_start_site+i+self.dog_updated_freq])
            critic_sum_rewards_for_update.append(critic_sum_reward)
            raw_sum_rewards_for_update.append(raw_sum_reward)

        input = torch.tensor(critic_sum_rewards_for_update, requires_grad=True)
        target = torch.tensor(raw_sum_rewards_for_update)
        critic_loss = self.critic_loss_fn(input, target)
        print("critic_loss:", critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def get_last_raw_rewards(self, nums):
        return self.raw_rewards[-nums:]

    def get_last_rewards(self, nums):
        if torch.is_tensor(self.rewards[0]):
            rewards_detached = [r.cpu().detach() for r in self.rewards[-nums:]]
            return rewards_detached
        else:
            return self.rewards[-nums:]

    def plot_rewards(self):
        plt.plot(self.log_rewards_means)
        plt.show()

    def save_critic(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.critic_net, model_path)

    def save_log(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(f"{filename}.json", "w") as f:
            json.dump({"raw_rewards_means": [float(r) for r in self.log_raw_rewards_means],
                       "rewards_means": [float(r) for r in self.log_rewards_means]}, f)
        plt.plot(self.log_raw_rewards_means)
        plt.savefig(f"{filename}_raw_rewards.png")
        plt.plot(self.log_rewards_means)
        plt.savefig(f"{filename}_rewards.png")
