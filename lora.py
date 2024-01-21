# this is code adapted from https://github.com/JamesQFreeman/LoRA-ViT

import torch.nn as nn

class LoRA_qkv(nn.Module):
    """ LoRA qkv module for Vision Transformer. """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.q_lora = nn.Sequential(linear_a_q, linear_b_q)
        self.v_lora = nn.Sequential(linear_a_v, linear_b_v)

    def forward(self, x):
        qkv = self.qkv(x) 
        new_q = self.q_lora(x)
        new_v = self.v_lora(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
