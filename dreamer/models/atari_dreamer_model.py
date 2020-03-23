import torch
import torch.nn as nn


class AtariDreamerModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, prev_action, prev_reward, init_rnn_state):
        return
