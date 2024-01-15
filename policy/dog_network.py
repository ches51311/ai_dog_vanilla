import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Recall:
    def __init__(self, nums, dim = 64, requires_grad=False):
        self.data = torch.zeros([nums,dim], requires_grad=requires_grad)
        self.nums = nums
        self.cnt = 0
    def save(self, t):
        self.data = self.data[:self.nums-1, :]
        self.data = torch.cat([t, self.data], dim = 0)
        self.cnt += 1
    def mature(self):
        return self.cnt >= self.nums
    def pick(self):
        self.cnt = 0
        return torch.mean(self.data, dim = 0, keepdim = True)

class RecallChunk:
    def __init__(self, dim = 64):
        self.dim = dim
        self.recall_10ms = Recall(100, dim = self.dim)
        self.recall_sec = Recall(60, dim = self.dim)
        self.recall_min = Recall(60, dim = self.dim)
        self.recall_hr = Recall(24, dim = self.dim)
        self.recall_day = Recall(100, dim = self.dim)
        # 344
        self.nums = self.recall_10ms.nums + \
                    self.recall_sec.nums + \
                    self.recall_min.nums + \
                    self.recall_hr.nums + \
                    self.recall_day.nums
    def save(self, recall):
        recall_clone = recall.clone().detach()
        recall_clone = recall_clone.reshape([1, self.dim])
        self.recall_10ms.save(recall_clone)
        if self.recall_10ms.mature():
            self.recall_sec.save(self.recall_10ms.pick())
        if self.recall_sec.mature():
            self.recall_min.save(self.recall_sec.pick())
        if self.recall_min.mature():
            self.recall_hr.save(self.recall_min.pick())
        if self.recall_hr.mature():
            self.recall_day.save(self.recall_hr.pick())
    @property
    def data(self):
        return torch.cat([self.recall_10ms.data, self.recall_sec.data,self.recall_min.data,
                             self.recall_hr.data, self.recall_day.data], dim = 0)

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

class DogNetworkBase(nn.Module):
    def __init__(self, backbone_output_size):
        super().__init__()
        self.state_size = 9 # dog_site breeder_site, hp, mp, life
        self.action_size = 5 # x, y, rotate, bark, shake
        self.eps = 1e-6

        # mean layer
        self.mean_net = nn.Sequential(nn.Linear(backbone_output_size, self.action_size), nn.Tanh())
        # std layer
        self.std_net = nn.Sequential(nn.Linear(backbone_output_size, self.action_size), nn.Tanh())

    def concat_states(self, dog_site, breeder_site, hp, mp, life):
        hp = (torch.Tensor([hp]).clip(min=-20, max=200)-90)/110
        mp = (torch.Tensor([mp]).clip(min=0, max=200)-100)/100
        life = (torch.Tensor([life]).clip(max=10000)-5000)/5000
        states = list(dog_site) + list(breeder_site) + [hp] + [mp] + [life]
        states = torch.tensor(states).reshape(self.state_size)
        return states.to(torch.float32)

        # states = list(dog_site) + list(breeder_site) + [hp] + [mp] + [life]
        # states = torch.tensor(states).reshape(self.state_size)
        # return states.to(torch.float32)


    def forward_after_backbone(self, backbone_out):
        action_means = self.mean_net(backbone_out)
        action_stddevs = torch.log(1 + torch.exp(self.std_net(backbone_out)))
        action_dist = Normal(action_means + self.eps, action_stddevs + self.eps)

        action = action_dist.sample()
        prob = action_dist.log_prob(action)
        
        return action.flatten(), prob

class DogNetworkLinear(DogNetworkBase):
    def __init__(self):
        self.hidden_feature_size = 128
        self.backbone_output_size = 128
        super().__init__(backbone_output_size = self.backbone_output_size)

        # self.backbone = nn.Sequential(
        #     nn.Linear(self.state_size, self.hidden_feature_size),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_feature_size, self.backbone_output_size),
        #     nn.ReLU(),
        # )
        self.block1 = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_feature_size),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(self.hidden_feature_size, self.backbone_output_size),
            nn.ReLU()
        )


    def backbone(self, states):
        hidden_feature1 = self.block1(states) # [345, 128]
        hidden_feature2 = self.block2(hidden_feature1)
        residual = hidden_feature1 + hidden_feature2
        return residual.flatten()

    def forward(self, states):
        backbone_out = self.backbone(states).flatten()
        action, prob = self.forward_after_backbone(backbone_out)
        action_clone = action.cpu().clone().numpy()
        #TODO: return should a tensor?
        return {"move": action_clone[:3],
                "bark": action_clone[3],
                "shake": action_clone[4],
                "prob": prob}

class DogNetworkLinearRecall(DogNetworkBase):
    def __init__(self):
        self.hidden_feature_size = 128
        self.backbone_output_size = 128
        super().__init__(backbone_output_size = self.backbone_output_size)

        self.recall_feature_size = self.state_size + self.action_size # 14
        self.saved_recalls = RecallChunk(dim = self.recall_feature_size)

        self.block1 = nn.Sequential(
            nn.Linear(self.recall_feature_size, self.hidden_feature_size),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(self.hidden_feature_size, self.backbone_output_size),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Linear(1 + self.saved_recalls.nums, 1),
            nn.ReLU(),
        )
        self.zero_action = torch.zeros(self.action_size)

    def backbone(self, recalls):
        hidden_feature1 = self.block1(recalls) # [345, 128]
        hidden_feature2 = self.block2(hidden_feature1)
        residual = hidden_feature1 + hidden_feature2
        residual = residual.transpose(1,0) # [128, 345]
        backbone_out = self.block3(residual) # [128, 1]
        return backbone_out.flatten()

    def forward(self, states):
        flash_recall_before_act = torch.cat(states, self.zero_action)
        recalls = torch.cat([flash_recall_before_act, self.saved_recalls.data]) # [345, 14]

        backbone_out = self.backbone(recalls)
        action, prob = self.forward_after_backbone(backbone_out)

        flash_recall_after_act = torch.cat([states, action])
        self.saved_recalls.save(flash_recall_after_act)

        action_clone = action.cpu().clone().numpy()
        return {"move": action_clone[:3],
                "bark": action_clone[3],
                "shake": action_clone[4],
                "prob": prob}

class DogNetworkTransformerRecall(DogNetworkBase):
    def __init__(self):
        self.recall_feature_size = self.state_size + self.action_size # 14
        #TODO: backbone output size 14 is to small, need to adjust the
        #transformer to let it 128
        self.backbone_output_size = self.recall_feature_size
        super().__init__(backbone_output_size = self.backbone_output_size)

        self.saved_recalls = RecallChunk(dim = self.recall_feature_size)

        self.positional_encoding = PostionalEncoding(d_model=self.recall_feature_size, dropout=0)
        #TODO: think detail of the transformer
        self.transformer = nn.Transformer(d_model=self.recall_feature_size,
                                nhead=1,
                                num_encoder_layers=2,
                                num_decoder_layers=2,
                                dim_feedforward=512,
                                batch_first=True)

        self.zero_action = torch.zeros(self.action_size)
    
    def backbone(self, recalls, flash_recall):
        recalls_pe = self.positional_encoding(recalls)
        backbone_out = self.transformer(src=recalls_pe, tgt=flash_recall)
        return backbone_out.flatten()

    def forward(self, states):
        flash_recall_before_act = torch.cat(states, self.zero_action)
        recalls = torch.cat([flash_recall_before_act, self.saved_recalls.data]) # [345, 14]
        backbone_out = self.backbone(recalls, flash_recall_before_act)
        action, prob = self.forward_after_backbone(backbone_out)

        flash_recall_after_act = torch.cat([states, action])
        self.saved_recalls.save(flash_recall_after_act)

        action_clone = action.cpu().clone().numpy()
        return {"move": action_clone[:3],
                "bark": action_clone[3],
                "shake": action_clone[4],
                "prob": prob}

class Critic(DogNetworkBase):
    '''
    estimate rewards sum of 100 steps 
    '''
    def __init__(self):
        super().__init__(1)
        self.hidden_feature_size = 128
        self.backbone = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_feature_size),
            nn.ReLU(),
            nn.Linear(self.hidden_feature_size, 1)
        )
        
    def forward(self, states):
        return self.backbone(states).flatten()
