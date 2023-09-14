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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Recall:
    def __init__(self, nums, dim = 64):
        self.data = torch.zeros([nums,dim]).detach().to(device)
        self.nums = nums
        self.cnt = 0
    def update(self, t):
        self.data = self.data[:self.nums-1, :]
        self.data = torch.cat([t, self.data], dim = 0)
        self.data.detach()
        self.cnt += 1
    def mature(self):
        return self.cnt >= self.nums
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


class DogNetworkTransformerRecall(nn.Module):
    def __init__(self):
        super(DogNetworkTransformerRecall,self).__init__()
        self.in_dim = 9
        self.out_dim = 5
        self.recall_dim = self.in_dim + self.out_dim
        self.positional_encoding = PostionalEncoding(d_model=self.recall_dim, dropout=0)
        self.t = nn.Transformer(d_model=self.recall_dim, nhead=1, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True)

        self.means = nn.Sequential(nn.Linear(self.recall_dim, self.out_dim), nn.Tanh())
        self.stds = nn.Sequential(nn.Linear(self.recall_dim, self.out_dim), nn.Tanh())

        self.recall_10ms = Recall(100, dim=self.recall_dim) # [100,64]
        self.recall_sec = Recall(60, dim=self.recall_dim)
        self.recall_min = Recall(60, dim=self.recall_dim)
        self.recall_hr = Recall(24, dim=self.recall_dim)
        self.recall_day = Recall(100, dim=self.recall_dim)

        self.eps = 1e-6

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
        dog_site = torch.Tensor(dog_site).to(device).reshape([3])
        man_site = torch.Tensor(man_site).to(device).reshape([3])
        hp = (torch.Tensor([hp]).to(device).reshape([1]).clip(min=-20, max=120)-50)/70
        mp = (torch.Tensor([mp]).to(device).reshape([1]).clip(min=-50, max=150)-50)/100
        life = (torch.Tensor([life]).to(device).reshape([1]).clip(max=10000)-5000)/5000
        concat = torch.cat([dog_site, man_site, hp, mp, life], dim=0).reshape([self.in_dim])
        cur_recall = nn.functional.pad(input=concat,  pad=(0,self.out_dim), mode='constant', value=0).reshape([1,1,self.recall_dim])

        recalls = torch.cat([self.recall_10ms.data, self.recall_sec.data,self.recall_min.data,
                             self.recall_hr.data, self.recall_day.data], dim = 0).reshape([1,-1,self.recall_dim])
        recalls_pe = self.positional_encoding(recalls)
        out = self.t(src=cur_recall, tgt=recalls_pe)#.reshape([1, self.recall_dim])
        out = out[:,-1,:].reshape([1, self.recall_dim])
        result_means = self.means(out).reshape([self.out_dim])
        result_stds = torch.log(1+torch.exp(self.stds(out))).reshape([self.out_dim])
        result_dist = Normal(result_means + self.eps, result_stds + self.eps)
        result = result_dist.sample()

        prob = result_dist.log_prob(result)

        # recall store input feature and action
        cur_recall = cur_recall.reshape([1,self.recall_dim]).clone()
        cur_recall[:,self.in_dim:] = result.reshape([1,self.out_dim])
        self.update_recall(cur_recall)

        result_clone = result.cpu().clone().detach().numpy()
        return {"move": result_clone[:3],
                "bark": result_clone[3],
                "shake": result_clone[4],
                "prob": prob}

class DogNetworkLinear(nn.Module):
    def __init__(self):
        super(DogNetworkLinear,self).__init__()
        in_dim = 9
        out_dim = 5
        hidden_space1 = 128  # Nothing special with 16, feel free to change
        hidden_space2 = 128  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net1 = nn.Sequential(
            nn.Linear(in_dim, hidden_space1),
            nn.ReLU()
        )
        self.shared_net2 = nn.Sequential(
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(nn.Linear(hidden_space2, out_dim), nn.Tanh())

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(nn.Linear(hidden_space2, out_dim), nn.Tanh())

        self.eps = 1e-6

    def forward(self, dog_site, man_site, hp, mp, life) -> Tuple[torch.Tensor, torch.Tensor]:
        dog_site = torch.Tensor(dog_site).to(device).reshape([3])
        man_site = torch.Tensor(man_site).to(device).reshape([3])
        hp = (torch.Tensor([hp]).to(device).reshape([1]).clip(min=-20, max=120)-50)/70
        mp = (torch.Tensor([mp]).to(device).reshape([1]).clip(min=-50, max=150)-50)/100
        life = (torch.Tensor([life]).to(device).reshape([1]).clip(max=1000)-500)/500
        x = torch.cat([dog_site, man_site, hp, mp, life], dim=0).reshape([1,9])
        shared_feature1 = self.shared_net1(x.float())
        shared_feature2 = self.shared_net2(shared_feature1)
        shared_features = shared_feature1 + shared_feature2

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action_clone = action.cpu().clone().detach().numpy()
        return {"move": action_clone[:3],
                "bark": action_clone[3],
                "shake": action_clone[4],
                "prob": prob}

class DogNetworkLinearRecall(nn.Module):
    def __init__(self):
        super(DogNetworkLinearRecall,self).__init__()

        in_dim = 9
        out_dim = 5
        hidden_space1 = 128  # Nothing special with 16, feel free to change
        hidden_space2 = 128  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net1 = nn.Sequential(
            nn.Linear(in_dim, hidden_space1),
            nn.ReLU()
        )
        self.shared_net2 = nn.Sequential(
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
        )

        self.recall_10ms = Recall(100, dim = hidden_space2) # [100,64]
        self.recall_sec = Recall(60, dim = hidden_space2)
        self.recall_min = Recall(60, dim = hidden_space2)
        self.recall_hr = Recall(24, dim = hidden_space2)
        self.recall_day = Recall(100, dim = hidden_space2)

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(nn.Linear(345*hidden_space2, out_dim), nn.Tanh())

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(nn.Linear(345*hidden_space2, out_dim), nn.Tanh())

        self.eps = 1e-6

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
        dog_site = torch.Tensor(dog_site).to(device).reshape([3])
        man_site = torch.Tensor(man_site).to(device).reshape([3])
        hp = (torch.Tensor([hp]).to(device).reshape([1]).clip(min=-20, max=120)-50)/70
        mp = (torch.Tensor([mp]).to(device).reshape([1]).clip(min=-50, max=150)-50)/100
        life = (torch.Tensor([life]).to(device).reshape([1]).clip(max=1000)-500)/500
        x = torch.cat([dog_site, man_site, hp, mp, life], dim=0).reshape([1,9])
        shared_feature1 = self.shared_net1(x.float())
        shared_feature2 = self.shared_net2(shared_feature1)
        shared_features = shared_feature1 + shared_feature2

        concat = torch.cat([shared_features, self.recall_10ms.data, self.recall_sec.data,self.recall_min.data,
                             self.recall_hr.data, self.recall_day.data], dim = 0).reshape([1,345*128])#.transpose(2,1)


        action_means = self.policy_mean_net(concat)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(concat))
        )
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        self.update_recall(shared_features)

        action_clone = action.cpu().clone().detach().numpy()
        return {"move": action_clone[:3],
                "bark": action_clone[3],
                "shake": action_clone[4],
                "prob": prob}


def summan_sai(dog_site, man_site, hp, mp, life) -> Tuple[torch.Tensor, torch.Tensor]:
    result = {}
    # keep close to npc
    distance = math.dist(man_site[:2], dog_site[:2])
    if distance < 1:
        result["move"] = np.array([0,0,0])
    else:
        x = man_site[0] - dog_site[0]
        y = man_site[1] - dog_site[1]
        s = abs(x) + abs(y)
        result["move"] = np.array([x*0.25/s, y*0.25/s, 0])
    # bark if hungry
    if hp < 80:
        result["bark"] = np.array([0.7])
    else:
        result["bark"] = np.array([0])
    # shake if bad mood
    if mp < 80:
        result["shake"] = np.array([0.7])
    else:
        result["shake"] = np.array([0])
    return result


class DogPolicy:
    def __init__(self, net_type = "linear", reward_type = "simple", enable_sai = False):
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

        if net_type == "linear":
            self.net = DogNetworkLinear().to(device)
            model_path = "dog_linear.pt"
        elif net_type == "linear_recall":
            self.net = DogNetworkLinearRecall().to(device)
            model_path = "dog_linear_recall.pt"
        elif net_type == "transformer_recall":
            self.net = DogNetworkTransformerRecall().to(device)
            model_path = "dog_transformer_recall.pt"
        else:
            assert(0 and "net_type illegal")
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path))
        self.enable_sai = enable_sai
        self.reward_type = reward_type
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.save_path = model_path
        self.log_path = "log.json"
        self.prev_distance = None
        self.rewards_mean_history = []
        self.update_cnt = 0

    def reborn(self, obs):
        obs["dog_action"] = {"move": np.zeros(3, np.float32),
                             "bark": 0,
                             "shake": 0,
                             "prob": None}
        self.hp = 100.
        self.mp = 100.
        if self.reward_type not in ["simple"]:
            self.rewards.append(-4444.+min(2000, self.life))
        self.life = 0.
        self.prev_distance = None
    
    def death(self):
        return self.hp < 0
    
    def metabolism(self, obs):
        distance = math.dist(obs["man_site"][:2], obs["dog_site"][:2])
        # man feed add hp
        if distance < 1 and obs["man_action"]["feed"] == True:
            print("feed!!")
            self.hp += 20
        # man touch add mp
        if distance < 1 and obs["man_action"]["touch"] == True:
            print("touch!!")
            self.mp += 20
        # # man near add mp
        # if distance < 0.5:
        #     print("near!!")
        #     self.mp += 5
        # if distance > 5:
        #     self.mp -= 1
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
        if len(self.rewards) < 100:
            return
        self.rewards_mean_history.append(sum(self.rewards)/len(self.rewards))
        print(self.update_cnt , ":mean(rewards):", self.rewards_mean_history[-1])
        # print(self.rewards)
        if self.probs == []:
            return

        running_g = 0
        gs = []
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # loss = 0
        # # minimize -1 * prob * reward obtained
        # for log_prob, reward in zip(self.probs, torch.tensor(self.rewards[1:])):
        #     if log_prob is None:
        #         continue
        #     loss += log_prob.mean() * reward * (-1)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.probs = []
        self.rewards = []
        self.update_cnt += 1
        if len(self.rewards_mean_history) % 500 == 0:
            self.save(self.save_path)
            self.save_log(self.log_path, show = False)

    def my_turn(self, obs):
        self.distance = math.dist(obs["man_site"][:2], obs["dog_site"][:2])
        # do action
        self.dog_action = self.net(obs["dog_site"], obs["man_site"], self.hp, self.mp, self.life)
        self.sai_action = summan_sai(obs["dog_site"], obs["man_site"], self.hp, self.mp, self.life)
        if self.enable_sai:
            obs["dog_action"] = self.sai_action
        else:
            obs["dog_action"] = self.dog_action

        self.probs.append(self.dog_action["prob"])

    def set_reward(self, obs):
        # update hp & mp
        self.metabolism(obs)

        # set reward & prob
        if self.enable_sai:
            reward = (sum(abs(self.dog_action["move"] - self.sai_action["move"])) +
                      sum(abs(self.dog_action["bark"] - self.sai_action["bark"])) +
                      sum(abs(self.dog_action["shake"] - self.sai_action["shake"]))) * -1
            self.rewards.append(reward)
        elif self.reward_type == "simple":            
            distance = math.dist(obs["man_site"][:2], obs["dog_site"][:2])
            reward = (self.distance - distance) * 1000
            self.rewards.append(reward)
        elif self.reward_type == "advance":
            reward = (abs(self.hp - 80) + abs(self.mp - 80)) * -1 + min(5000, self.life * 0.001)
            self.rewards.append(reward)
        else:
            assert(0 and "reward_type illegal")

    def save(self, model_path = ""):
        if model_path != "":
            torch.save(self.net.state_dict(), model_path)
    
    def save_log(self, path = "log.json", show = True):
        with open(path, "w") as f:
            json.dump({"rewards_mean_history": self.rewards_mean_history}, f)
        plt.plot(self.rewards_mean_history)
        plt.savefig(path + ".png")

    def plot_rewards(self):
        plt.plot(self.rewards_mean_history)
        plt.show()
