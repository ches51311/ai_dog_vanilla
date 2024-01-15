from dog_network import *
import numpy as np
import torch
import torch.onnx
import os
import math
import json
import copy
import matplotlib.pyplot as plt

torch.manual_seed(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def summan_sai(dog_site, breeder_site, hp, mp, life):
    result = {}
    # keep close to npc
    distance = math.dist(breeder_site[:2], dog_site[:2])
    if distance < 1:
        result["move"] = np.array([0,0,0])
    else:
        x = breeder_site[0] - dog_site[0]
        y = breeder_site[1] - dog_site[1]
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
    def __init__(self, dog_reward, net_type = "linear", saved_dir = os.path.join(os.getcwd(), "models")):
        self.dog_reward = dog_reward
        self.net_type = net_type
        self.saved_dir = saved_dir

        self.set_model()
        self.learning_rate = 1e-4
        self.gamma = 0.9
        self.eps = 1e-6
        self.optimizer = torch.optim.AdamW(self.dog_net.parameters(), lr=self.learning_rate)
        self.update_weight_time = 100
        self.update_weight_cnt = 0

        self.saved_actions = []
        self.death_action = {"move": np.zeros(3, np.float32),
                             "bark": 0,
                             "shake": 0,
                             "prob": None}

    def set_model(self):
        self.dog_model_path = os.path.join(self.saved_dir, f"dog_{self.net_type}.pth")
        if os.path.exists(self.dog_model_path):
            self.dog_net = torch.load(self.dog_model_path)
        elif self.net_type == "linear":
            self.dog_net = DogNetworkLinear().to(device)
        elif self.net_type == "linear_recall":
            self.dog_net = DogNetworkLinearRecall().to(device)
        elif self.net_type == "transformer_recall":
            self.dog_net = DogNetworkTransformerRecall().to(device)
        else:
            assert(0 and "net_type illegal")

        self.dog_net.train()

    def my_turn(self, obs):
        if len(self.saved_actions) == self.update_weight_time:
            self.update_weight()
        
        dog_vital_sign = self.dog_reward.dog_vital_sign
        if dog_vital_sign.state in ["just_death", "death"]:
            obs["dog_action"]  = self.death_action
        else:
            states = self.dog_net.concat_states(obs["dog_site"], obs["breeder_site"], \
                                      dog_vital_sign.hp, dog_vital_sign.mp, dog_vital_sign.life)
            dog_action = self.dog_net(states.to(device))
            obs["dog_action"] = dog_action
            self.saved_actions.append(dog_action)

    def update_weight(self):
        probs = [action["prob"] for action in self.saved_actions]
        rewards = self.dog_reward.get_rewards()#[-len(probs):]

        running_g = 0
        gs = []
        for R in rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0

        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.saved_actions = []


        self.update_weight_cnt += 1
        if self.update_weight_cnt % 500 == 0:
            self.save_dog(self.dog_model_path)
            # self.dog_reward.save_rewards()

    def save_dog(self, model_path = ""):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if model_path != "":
            torch.save(self.dog_net, model_path)
