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
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 32  # Nothing special with 16, feel free to change
        hidden_space2 = 64  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs

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

        self.net = DogNetwork(dog_obs_dims, dog_action_dims)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path))
        self.use_cuda = use_cuda
        if self.use_cuda == True:
            self.net.to('cuda')
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
    
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
        self.hp -= 0.001
        self.mp -= 0.01

        # 2. set reward
        reward = (abs(self.hp - 80) + abs(self.mp - 80)) * -1 + self.life * 0.001
        self.rewards.append(reward)

        # 3. update weight
        # self.update_weight()
        # 4. do action
        action = {}
        # dog_obs = torch.tensor(np.array([dog_obs]))
        # if self.use_cuda == True:
        #     dog_obs = dog_obs.to('cuda')
        # move_means, move_stddevs = self.net(dog_obs)

        # # create a normal distribution from the predicted
        # #   mean and standard deviation and sample an action
        # distrib = Normal(move_means[0] + self.eps, move_stddevs[0] + self.eps)
        # move = distrib.sample()
        # prob = distrib.log_prob(move)
        # self.probs.append(prob)

        # move = move.to('cpu').numpy()
        # action["move"] = np.concatenate([move, np.zeros(1)])
        action["move"] = np.random.uniform(-0.5,0.5,3)
        action["bark"] = np.random.uniform(0,0.5,1)
        action["shake"] = np.random.uniform(0,0.5,1)

        obs["dog_action"] = action

    def save(self, model_path = ""):
        if model_path != "":
            torch.save(self.net.state_dict(), model_path)
