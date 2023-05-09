from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import Tuple
import torch.onnx

torch.manual_seed(0)

# class Policy_Network(nn.Module):
#     """Parametrized Policy Network."""

#     def __init__(self, obs_space_dims: int, action_space_dims: int):
#         """Initializes a neural network that estimates the mean and standard deviation
#          of a normal distribution from which an action is sampled from.

#         Args:
#             obs_space_dims: Dimension of the observation space
#             action_space_dims: Dimension of the action space
#         """
#         super().__init__()

#         hidden_space1 = 32  # Nothing special with 16, feel free to change
#         hidden_space2 = 64  # Nothing special with 32, feel free to change

#         # Shared Network
#         self.shared_net1 = nn.Sequential(
#             nn.Linear(obs_space_dims, hidden_space1),
#             nn.Tanh(),
#         )

#         self.shared_net2 = nn.Sequential(
#             nn.Linear(hidden_space1, hidden_space2),
#             nn.Tanh(),
#         )


#         # Policy Mean specific Linear Layer
#         self.policy_mean_net = nn.Sequential(
#             nn.Linear(hidden_space2, action_space_dims)
#         )

#         # Policy Std Dev specific Linear Layer
#         self.policy_stddev_net = nn.Sequential(
#             nn.Linear(hidden_space2, action_space_dims)
#         )

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Conditioned on the observation, returns the mean and standard deviation
#          of a normal distribution from which an action is sampled from.

#         Args:
#             x: Observation from the environment

#         Returns:
#             action_means: predicted mean of the normal distribution
#             action_stddevs: predicted standard deviation of the normal distribution
#         """
#         shared_features1_1 = self.shared_net1(x.float())
#         shared_features1_2 = self.shared_net1(x.float())
#         sub_feature = torch.sub(shared_features1_1, shared_features1_2)
#         shared_features2 = self.shared_net2(sub_feature)
#         action_means = self.policy_mean_net(shared_features2)
#         action_stddevs = torch.log(
#             1 + torch.exp(self.policy_stddev_net(shared_features2))
#         )

#         return action_means, action_stddevs

class Policy_Network(nn.Module):
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

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.8  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.losses = []

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action
    def update(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) > 1:
            actual_reward = self.rewards[-2] - self.rewards[-1]
            delta = torch.tensor(actual_reward)
            loss = self.probs[-1].mean() * delta * (-1)
            self.losses.append(loss.detach().numpy())
            # if len(self.rewards) % 10 == 0:
            #     print("loss:", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    def clean(self):
        self.probs = []
        self.rewards = []
        self.losses = []
    # def update(self):
    #     """Updates the policy network's weights."""
    #     if len(self.rewards) <= 1:
    #         return
    #     loss = 0
    #     for log_prob in self.probs:
    #         loss += log_prob.mean()
    #     loss /= len(self.probs)
    #     loss = abs(loss) + self.eps
    #     loss *= self.rewards[-1] * (-1)
    #     self.loss = loss.item()
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # Empty / zero out all episode-centric/related variables
    #     self.probs = []
    #     self.rewards = []

    # def update(self):
    #     """Updates the policy network's weights."""
    #     # self.rewards is storing distance
    #     actual_rewards = []
    #     for i in range(len(self.rewards)-1):
    #         actual_rewards.append((self.rewards[i] - self.rewards[i-1])*1000/len(self.rewards))
    #     # print("actual rewards:", actual_rewards)
    #     deltas = torch.tensor(actual_rewards)

    #     loss = 0
    #     # minimize -1 * prob * reward obtained
    #     for log_prob, delta in zip(self.probs, deltas):
    #         loss += log_prob.mean() * delta * (-1)
    #     if type(loss) == torch.Tensor:
    #         self.loss = loss.item()
    #         # Update the policy network
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #     else:
    #         self.loss = loss

    #     # Empty / zero out all episode-centric/related variables
    #     self.probs = []
    #     self.rewards = []
    def save(self, input_shape):
        x = torch.randn(input_shape, requires_grad=True)
        torch.onnx.export(self.net,               # model being run
                          x,
                        "dog.onnx",   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        )



class DogPolicy:
    def __init__(self, ob_space, ac_space):
        self.ob_space = ob_space
        self.ac_space = ac_space
        #TODO: self.ob_space.shape[0] is 16, but only need xy of npc & dog
        #TODO: -4 is to remove npc's space & dog's rotate
        self.dog = REINFORCE(4, self.ac_space.shape[0]-4)
    def add_reward(self, reward):
        self.dog.rewards.append(reward)
    def update(self, reward):
        self.dog.update(reward)
    def clean_reward(self):
        self.dog.clean()
    def update_dog(self):
        self.dog.update()

    def act(self, obs):
        return self.dog.sample_action(obs)
    def save(self):
        self.dog.save(4)
