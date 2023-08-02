from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import Tuple
import torch.onnx
import os
import math
import json
import matplotlib.pyplot as plt

torch.manual_seed(3)

class Recall:
    def __init__(self, nums, dim = 14):
        self.data = torch.zeros([nums,dim,1]).detach()
        self.nums = nums
        self.cnt = 0
    def update(self, t):
        self.data = self.data[:self.nums-1, :]
        self.data = torch.cat([t, self.data], dim = 0)
        self.data.detach()
        self.cnt += 1
    def mature(self):
        return self.cnt == self.nums
    def pick(self):
        self.cnt = 0
        return torch.mean(self.data, dim = 0, keepdim = True)

class PostionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PostionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        x = x + self.pe[:,:x.size(1)].requires_grad_(False)
        return self.dropout(x)


class DogNetwork(nn.Module):
    def __init__(self, no_cuda = True):
        super(DogNetwork,self).__init__()
        self.no_cuda = no_cuda
        self.l1 = nn.Linear(1, 64)
        self.positional_encoding = PostionalEncoding(d_model=64, dropout=0)
        self.t1 = nn.TransformerEncoderLayer(d_model=64, nhead=1, batch_first=True)
        self.t2 = nn.TransformerEncoderLayer(d_model=345, nhead=1, batch_first=True)

        self.means = nn.Sequential(nn.Linear(14*345, 5), nn.Tanh())
        self.stds = nn.Sequential(nn.Linear(14*345, 5), nn.Tanh())

        self.recall_10ms = Recall(100)
        self.recall_sec = Recall(60)
        self.recall_min = Recall(60)
        self.recall_hr = Recall(24)
        self.recall_day = Recall(100)

        self.eps = 1e-6

        if not no_cuda:
            self.t1 = self.t1.cuda()
            self.t2 = self.t2.cuda()
            self.means = self.means.cuda()
            self.stds = self.stds.cuda()
            self.recall_10ms.data = self.recall_10ms.data.cuda()
            self.recall_sec.data = self.recall_sec.data.cuda()
            self.recall_min.data = self.recall_min.data.cuda()
            self.recall_hr.data = self.recall_hr.data.cuda()
            self.recall_day.data = self.recall_day.data.cuda()

    def update_recall(self, recall):
        recall_clone = recall.clone().detach()
        self.recall_10ms.update(recall_clone)
        if self.recall_10ms.mature():
            self.recall_sec.update(self.recall_10ms.pick())
        if self.recall_sec.mature():
            self.recall_min.update(self.recall_sec.pick())
        if self.recall_min.mature():
            self.recall_hr.update(self.recall_min.pick())
        if self.recall_hr.mature():
            self.recall_day.update(self.recall_hr.pick())

    def forward(self, dog_site, man_site, hp, mp, life) -> Tuple[torch.Tensor, torch.Tensor]:
        dog_site = torch.Tensor(dog_site).reshape([3,1])
        man_site = torch.Tensor(man_site).reshape([3,1])
        hp = torch.Tensor([hp]).reshape([1,1])
        mp = torch.Tensor([mp]).reshape([1,1])
        life = torch.Tensor([life]).reshape([1,1])
        if not self.no_cuda:
            dog_site = dog_site.cuda()
            man_site = man_site.cuda()
            hp = hp.cuda()
            mp = mp.cuda()
            life = life.cuda()
        concat = torch.cat([dog_site, man_site, hp, mp, life], dim=0)
        linear = self.l1(concat).reshape([1,9,64])
        pe = self.positional_encoding(linear)
        pe_t = self.t1(pe)
        pe_t_mean = torch.mean(pe_t, dim=2, keepdim=True)
        recall = nn.functional.pad(input=pe_t_mean,  pad=(0,0,0,5), mode='constant', value=0)
        concat2 = torch.cat([recall, self.recall_10ms.data, self.recall_sec.data,self.recall_min.data,
                             self.recall_hr.data, self.recall_day.data], dim = 0).reshape([1,345,14]).transpose(2,1)
        concat2_t = self.t2(concat2).reshape([1, 14*345])
        result_means = self.means(concat2_t).reshape([5])
        result_stds = torch.log(1+torch.exp(self.stds(concat2_t))).reshape([5])
        result_dist = Normal(result_means + self.eps, result_stds + self.eps)
        result = result_dist.sample()

        prob = result_dist.log_prob(result)

        # recall store input feature and action
        recall[:,9:,:] = result.reshape([5,1])
        self.update_recall(recall)

        result_clone = result.cpu().clone().detach().numpy()
        return {"move": result_clone[:3],
                "bark": result_clone[3],
                "shake": result_clone[4],
                "prob": prob}

class DogPolicy:
    def __init__(self, model_path = "", no_cuda = True, auto_save = True, save_path = "dog.pt"):
        # dog's hungry point & mood point
        self.hp = 100.
        self.mp = 100.
        self.life = 0.
        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.8  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = DogNetwork(no_cuda)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path))
        self.no_cuda = no_cuda
        if not self.no_cuda:
            self.net.to('cuda')
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.auto_save = auto_save
        self.save_path = save_path
        self.rewards_mean_history = []
    
    def reborn(self, obs):
        obs["dog_action"] = {"move": np.zeros(3, np.float32),
                             "bark": 0,
                             "shake": 0,
                             "prob": None}
        self.hp = 100.
        self.mp = 100.
        self.rewards.append(-4444.+min(2000, self.life))
        self.life = 0.
    
    def death(self):
        return self.hp < 0
    
    def metabolism(self, obs):
        distance = math.dist(obs["man_site"][:2], obs["dog_site"][:2])
        # man feed add hp
        if distance < 0.3 and obs["man_action"]["feed"] == True:
            print("feed!!")
            self.hp += 20
        # man touch add mp
        if distance < 0.3 and obs["man_action"]["touch"] == True:
            print("touch!!")
            self.mp += 20
        # man near add mp
        if distance < 0.3:
            print("near!!")
            self.mp += 5
        if distance > 5:
            self.hp -= 1
            self.mp -= 1
        # dog's action influence hp/mp
        dog_momentum = pow(pow(obs["dog_action"]["move"][0], 2) + \
                           pow(obs["dog_action"]["move"][1], 2), 1/2) + abs(obs["dog_action"]["move"][2])
        self.hp -= dog_momentum * 0.01
        self.hp -= abs(obs["dog_action"]["bark"]) * 0.01
        self.hp -= abs(obs["dog_action"]["shake"]) * 0.01
        self.life += 1
        self.hp -= 0.01
        self.mp -= 0.1
        self.mp = max(0., self.mp)

    def update_weight(self):
        self.rewards_mean_history.append(sum(self.rewards)/len(self.rewards))
        print("mean(rewards):", self.rewards_mean_history[-1])
        if self.probs == []:
            return
        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, reward in zip(self.probs, torch.tensor(self.rewards)):
            if log_prob is None:
                continue
            loss += log_prob.mean() * reward * (-1)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.probs = []
        self.rewards = []
        if self.auto_save and len(self.rewards_mean_history) % 500 == 0:
            self.save(self.save_path)

    def my_turn(self, obs):
        # 1. update hp & mp
        self.metabolism(obs)

        # 2. set reward & prob
        reward = (abs(self.hp - 80) + abs(self.mp - 80)) * -1 + min(200, self.life * 0.01)
        self.rewards.append(reward)
        self.probs.append(obs["dog_action"]["prob"])
    
        # 3. update weight
        if len(self.rewards) > 100:
            self.update_weight()

        # 4. do action
        obs["dog_action"] = self.net(obs["dog_site"], obs["man_site"], self.hp, self.mp, self.life)


    def save(self, model_path = ""):
        if model_path != "":
            torch.save(self.net.state_dict(), model_path)
    
    def save_reward_history(self, path):
        with open(path, "w") as f:
            json.dump({"rewards_mean_history": self.rewards_mean_history}, f)
        plt.plot(self.rewards_mean_history)
        plt.savefig(path + ".png")
        plt.show()
