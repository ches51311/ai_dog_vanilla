import os
import math
import json
import torch
import matplotlib.pyplot as plt
from dog_network import *

torch.manual_seed(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DogReward:
    def __init__(self, dog_vital_sign, reward_type = "simple", saved_dir = os.path.join(os.getcwd(), "models")) -> None:
        self.dog_vital_sign = dog_vital_sign
        self.reward_type = reward_type
        self.saved_dir = saved_dir

        self.set_model()
        self.learning_rate = 1e-4
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=self.learning_rate) # dog_reward
        self.critic_loss_fn = torch.nn.SmoothL1Loss()
        self.update_critic_weight_time = 100
        self.update_critic_cnt = 0

        self.prev_distance = None
        self.prev_obs = None
        self.raw_rewards = []
        self.rewards = []
        self.raw_rewards_mean = []
        self.rewards_mean = []

    def set_model(self):
        self.critic_model_path = os.path.join(self.saved_dir, f"critic_{self.reward_type}.pth")
        if os.path.exists(self.critic_model_path):
            self.critic_net = torch.load(self.critic_model_path)
        else:
            self.critic_net = Critic().to(device)
        self.critic_net.train()

    def my_turn(self, obs):
        if len(self.rewards) == self.update_critic_weight_time:
            self.update_critic_weight()

        if self.dog_vital_sign.state in ["death", "just_reborn"]:
            return
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

        if self.prev_obs is None:
            self.prev_obs = obs
        prev_states = self.critic_net.concat_states(self.prev_obs["dog_site"], self.prev_obs["breeder_site"], \
                                              self.dog_vital_sign.prev_hp, self.dog_vital_sign.prev_mp, self.dog_vital_sign.prev_life)
        self.prev_critic_reward = self.critic_net(prev_states.to(device))
        states = self.critic_net.concat_states(obs["dog_site"], obs["breeder_site"], \
                                         self.dog_vital_sign.hp, self.dog_vital_sign.mp, self.dog_vital_sign.life)
        self.critic_reward = self.critic_net(states.to(device))
        reward = raw_reward + self.critic_reward * 0.99 - self.prev_critic_reward

        self.raw_rewards.append(raw_reward)
        self.rewards.append(reward)
        self.prev_obs = obs

    def update_critic_weight(self):
        input = torch.tensor(self.rewards, requires_grad=True)
        target = torch.tensor(self.raw_rewards)
        critic_loss = self.critic_loss_fn(input, target).sum()
        print("critic_loss:", critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.raw_rewards_mean.append(torch.tensor(self.raw_rewards).float().mean())
        self.rewards_mean.append(torch.tensor(self.rewards).float().mean())
        self.raw_rewards = []
        self.rewards = []

        print(self.update_critic_cnt , ":mean(raw_rewards):", self.raw_rewards_mean[-1])
        print(self.update_critic_cnt , ":mean(rewards):", self.rewards_mean[-1])
        self.update_critic_cnt += 1
        if self.update_critic_cnt % 500 == 0:
            self.save_critic(self.critic_model_path)
            self.save_rewards()

    def get_rewards(self):
        rewards_detached = [r.cpu().detach() for r in self.rewards]
        return rewards_detached

    def plot_rewards(self):
        plt.plot(self.rewards_mean)
        plt.show()

    def save_critic(self, model_path = ""):
        if model_path != "":
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.critic_net, model_path)

    def save_rewards(self):
        with open("rewards.json", "w") as f:
            json.dump({"raw_rewards_mean": [float(r) for r in self.raw_rewards_mean],
                       "rewards_mean": [float(r) for r in self.rewards_mean]}, f)
        plt.plot(self.raw_rewards_mean)
        plt.savefig("raw_rewards.png")
        plt.plot(self.rewards_mean)
        plt.savefig("rewards.png")
