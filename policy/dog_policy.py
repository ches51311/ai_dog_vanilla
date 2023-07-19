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
    def __init__(self, use_cuda = False):
        super(DogNetwork,self).__init__()
        self.use_cuda = use_cuda
        self.t1 = nn.Transformer(d_model = 3, nhead=1, num_encoder_layers = 1, num_decoder_layers = 1, dim_feedforward = 256)
        self.t2 = nn.Transformer(d_model = 1, nhead=1, num_encoder_layers = 1, num_decoder_layers = 1, dim_feedforward = 256)
        self.t3 = nn.Transformer(d_model = 8, nhead=1, num_encoder_layers = 1, num_decoder_layers = 1, dim_feedforward = 256)
        self.l1 = nn.Linear(8, 3)
        self.l2 = nn.Linear(8, 1)

        if use_cuda:
            self.t1 = self.t1.cuda()
            self.t2 = self.t2.cuda()
            self.t3 = self.t3.cuda()
            self.l1 = self.l1.cuda()
            self.l2 = self.l2.cuda()

    def forward(self, dog_site, man_site, hp, mp) -> Tuple[torch.Tensor, torch.Tensor]:
        dog_site = torch.Tensor(dog_site).reshape([1,1,3])
        man_site = torch.Tensor(man_site).reshape([1,1,3])
        hp = torch.Tensor([hp]).reshape([1,1,1])
        mp = torch.Tensor([mp]).reshape([1,1,1])
        if self.use_cuda:
            dog_site = dog_site.cuda()
            man_site = man_site.cuda()
            hp = hp.cuda()
            mp = mp.cuda()

        dog_site_t = self.t1(dog_site, dog_site)
        man_site_t = self.t1(man_site, man_site)
        hp_t = self.t2(hp, hp)
        mp_t = self.t2(mp, mp)
        concat = torch.cat([dog_site_t, man_site_t, hp_t, mp_t], dim=2)
        q = concat
        result_move = self.t3(q,q).reshape([1,8])
        result_move = self.l1(result_move).reshape([3]).cpu().detach().numpy()
        result_bark = self.t3(q,q).reshape([1,8])
        result_bark = self.l2(result_bark).reshape([1]).cpu().detach().numpy()
        result_shake = self.t3(q,q).reshape([1,8])
        result_shake = self.l2(result_shake).reshape([1]).cpu().detach().numpy()

        return {"move": result_move, "bark": result_bark, "shake": result_shake}

    def backward(self):
        pass


class DogPolicy:
    def __init__(self, dog_obs_dims: int, dog_action_dims: int, model_path = "", use_cuda = False):
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

        self.net = DogNetwork(use_cuda)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path))
        self.use_cuda = use_cuda
        if self.use_cuda == True:
            self.net.to('cuda')
        # self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
    
    def reborn(self):
        self.hp = 100.
        self.mp = 100.
        self.life = 0.
        self.rewards.append(-100.)
    
    def death(self):
        return self.hp < 0

    def update_weight(self):
        # conditionally update weight

        """Updates the policy network's weights."""
        # self.rewards is storing distance
        actual_rewards = []
        for i in range(len(self.rewards)-1):
            actual_rewards.append((self.rewards[i] - self.rewards[i+1])*1000/len(self.rewards))
        # print("actual rewards:", actual_rewards)
        self.actual_reward_sum = sum(actual_rewards)
        deltas = torch.tensor(actual_rewards)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)
        if type(loss) == torch.Tensor:
            self.loss = loss.item()
            # Update the policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.loss = loss

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


    def my_turn(self, obs):
        # 1. update hp & mp
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
        # dog action influence hp/mp
        dog_momentum = pow(pow(obs["dog_action"]["move"][0], 2) + \
                           pow(obs["dog_action"]["move"][1], 2), 1/2) + obs["dog_action"]["move"][2]
        self.hp -= dog_momentum * 0.01
        self.hp -= obs["dog_action"]["bark"] * 0.01
        self.hp -= obs["dog_action"]["shake"] * 0.01
        self.life += 1
        self.hp -= 0.1
        self.mp -= 0.01

        # 2. set reward
        reward = (abs(self.hp - 80) + abs(self.mp - 80)) * -1 + self.life * 0.001
        self.rewards.append(reward)

        obs["dog_action"] = self.net(obs["dog_site"], obs["man_site"], self.hp, self.mp)

    def save(self, model_path = ""):
        if model_path != "":
            torch.save(self.net.state_dict(), model_path)
