from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import Tuple
import torch.onnx
import os
import math

torch.manual_seed(3)

class DogNetwork(nn.Module):
    def __init__(self, no_cuda = True):
        super(DogNetwork,self).__init__()
        self.no_cuda = no_cuda
        self.t1 = nn.TransformerEncoderLayer(d_model=3, nhead=1)
        self.t2 = nn.TransformerEncoderLayer(d_model=1, nhead=1)
        self.t3 = nn.TransformerEncoderLayer(d_model=8, nhead=1)
        self.l1 = nn.Sequential(nn.Linear(8, 3), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(8, 1), nn.Tanh())

        self.recall = torch.zeros([1,100,8]).detach()
        self.eps = 1e-6

        if not no_cuda:
            self.t1 = self.t1.cuda()
            self.t2 = self.t2.cuda()
            self.t3 = self.t3.cuda()
            self.l1 = self.l1.cuda()
            self.l2 = self.l2.cuda()
            self.recall = self.recall.cuda()

    def forward(self, dog_site, man_site, hp, mp) -> Tuple[torch.Tensor, torch.Tensor]:
        dog_site = torch.Tensor(dog_site).reshape([1,1,3])
        man_site = torch.Tensor(man_site).reshape([1,1,3])
        hp = torch.Tensor([hp]).reshape([1,1,1])
        mp = torch.Tensor([mp]).reshape([1,1,1])
        if not self.no_cuda:
            dog_site = dog_site.cuda()
            man_site = man_site.cuda()
            hp = hp.cuda()
            mp = mp.cuda()

        dog_site_t = self.t1(dog_site)
        man_site_t = self.t1(man_site)
        hp_t = self.t2(hp)
        mp_t = self.t2(mp)

        concat = torch.cat([dog_site_t, man_site_t, hp_t, mp_t], dim=2)
        self.recall = self.recall[:, :99].detach()
        self.recall = torch.cat([concat, self.recall], dim = 1)
        recall_t = self.t3(self.recall)
        recall_sum = torch.sum(recall_t, dim = 1)

        result_move_mean = self.l1(recall_sum).reshape([3])
        result_move_std = self.l1(recall_sum).reshape([3])
        result_move_std = torch.log(1+torch.exp(result_move_std))
        result_move_dist = Normal(result_move_mean + self.eps, result_move_std + self.eps)
        result_move = result_move_dist.sample()

        result_bark_mean = self.l2(recall_sum).reshape([1])
        result_bark_std = self.l2(recall_sum).reshape([1])
        result_bark_std = torch.log(1+torch.exp(result_bark_std))
        result_bark_dist = Normal(result_bark_mean + self.eps, result_bark_std + self.eps)
        result_bark = result_bark_dist.sample()

        result_shake_mean = self.l2(recall_sum).reshape([1])
        result_shake_std = self.l2(recall_sum).reshape([1])
        result_shake_std = torch.log(1+torch.exp(result_shake_std))
        result_shake_dist = Normal(result_shake_mean + self.eps, result_shake_std + self.eps)
        result_shake = result_shake_dist.sample()

        prob = torch.cat([result_move_dist.log_prob(result_move), 
                          result_bark_dist.log_prob(result_bark),
                          result_shake_dist.log_prob(result_shake)])

        clip = lambda x: x if x > 0 else [0.]
        return {"move": result_move.cpu().detach().numpy(),
                "bark": clip(result_bark.cpu().detach().numpy())[0],
                "shake": clip(result_shake.cpu().detach().numpy())[0],
                "prob": prob}

class DogPolicy:
    def __init__(self, model_path = "", no_cuda = True):
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
    
    def reborn(self, obs):
        obs["dog_action"] = {"move": np.zeros(3, np.float32),
                             "bark": 0,
                             "shake": 0,
                             "prob": None}
        self.hp = 100.
        self.mp = 100.
        self.life = 0.
        self.rewards.append(-100.)
    
    def death(self):
        return self.hp < 0
    
    def metabolism(self, obs):
        distance = math.dist(obs["man_site"][:2], obs["dog_site"][:2])
        # man feed add hp
        if distance < 0.3 and obs["man_action"]["feed"] == True:
            self.hp += 20
        # man touch add mp
        if distance < 0.3 and obs["man_action"]["touch"] == True:
            self.mp += 20
        # man near add mp
        if distance < 0.3:
            self.mp += 0.001
        # dog's action influence hp/mp
        dog_momentum = pow(pow(obs["dog_action"]["move"][0], 2) + \
                           pow(obs["dog_action"]["move"][1], 2), 1/2) + abs(obs["dog_action"]["move"][2])
        self.hp -= dog_momentum * 0.1
        self.hp -= abs(obs["dog_action"]["bark"]) * 0.01
        self.hp -= abs(obs["dog_action"]["shake"]) * 0.01
        self.life += 1
        self.hp -= 0.1
        self.mp -= 0.01

    def update_weight(self):
        if self.probs == []:
            return
        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, reward in zip(self.probs, torch.tensor(self.rewards)):
            if log_prob is None:
                continue
            loss += log_prob.mean() * reward * (-1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []

    def my_turn(self, obs):
        # 1. update hp & mp
        self.metabolism(obs)

        # 2. set reward
        reward = (abs(self.hp - 80) + abs(self.mp - 80)) * -1 + self.life * 0.001
        self.rewards.append(reward)
        self.probs.append(obs["dog_action"]["prob"])
    
        # 3. update weight
        if self.life % 100 == 0:
            self.update_weight()

        # 4. do action
        obs["dog_action"] = self.net(obs["dog_site"], obs["man_site"], self.hp, self.mp)


    def save(self, model_path = ""):
        if model_path != "":
            torch.save(self.net.state_dict(), model_path)
