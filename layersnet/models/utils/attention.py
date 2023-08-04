import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList
from layersnet.models.utils import FFN, GroupFFN

from .rotation_layer import RotFFN
from ..builder import ATTENTION
from .norm_func import attn_rotation


@ATTENTION.register_module()
class RotAttention(BaseModule):
    def __init__(self, dim, num_heads=8, dropout=0.0, 
                 act_cfg=dict(type='ReLU', inplace=False), eps=1e-7, **kwargs):
        super(RotAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.eps = eps

        self.dropout = nn.Dropout(dropout)
        self.r_activate = nn.Identity()
        self.s_activate = nn.Identity()
        self.m_activate = nn.Identity()
        
        self.receiver_weight = nn.Linear(dim, dim, bias=False)
        self.sender_weight = nn.Linear(dim, dim, bias=False)
        self.memory_weight = nn.Linear(dim, dim, bias=False)
        
        self.q_proj = FFN([dim, dim], final_act=False, bias=False, act_cfg=act_cfg)
        self.norms = ModuleList()
        self.norms.append(nn.Identity())
        self.norms.append(nn.Identity())
        self.norms.append(nn.Identity())
        self.proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)
        self.r_proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)
        self.s_proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)

        # For normalizations
        self.eps = eps
        self.n_weight = nn.Parameter(torch.Tensor(1, 1, dim))
        self.n_bias = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.ones_(self.n_weight)
        nn.init.zeros_(self.n_bias)
        
        head_dim = dim//self.num_heads
        mask_type = 2
        self.rot_mapping = RotFFN([3, head_dim], num_groups=self.num_heads//mask_type, final_act=False, bias=False, act_cfg=act_cfg)
        self.rs_mlp = GroupFFN([head_dim, head_dim], num_groups=self.num_heads//mask_type, final_act=True, bias=True, act_cfg=act_cfg)

    def forward(self,
            x_state, x_receiver, x_sender,
            h_state, h_sender,
            h_rotation,
            rs_mask=None, h_rs_mask=None,
            # Residual parts
            residual_r=None, residual_s=None, residual_hs=None,
            **kwargs):
        B, N, C = x_state.shape
        _, _, num_neighbor, _ = h_sender.shape
        N_r = x_receiver.shape[1]
        N_s = x_sender.shape[1]
        N_hs = h_rotation.shape[1]
        mask_type = rs_mask.shape[1]
        head_dim = C // self.num_heads            

        r_mem = self.m_activate(self.memory_weight(x_receiver))
        s_mem = self.m_activate(self.memory_weight(x_sender))
        hs_mem = self.m_activate(self.memory_weight(h_sender))
        x_receiver = self.r_activate(self.receiver_weight(x_state[:, :N_r])) + r_mem
        x_sender = self.s_activate(self.sender_weight(x_state)) + s_mem
        h_sender = self.s_activate(self.sender_weight(h_state)) + hs_mem
        
        # Residual
        x_receiver += residual_r
        x_sender += residual_s
        h_sender += residual_hs

        x_receiver = x_receiver.reshape(B, N_r, self.num_heads, head_dim).permute(0, 2, 1, 3)
        x_receiver = x_receiver.reshape(B, self.num_heads // mask_type, mask_type, N_r, head_dim)
        x_sender = x_sender.reshape(B, N_s, self.num_heads, head_dim).permute(0, 2, 1, 3)
        x_sender = x_sender.reshape(B, self.num_heads // mask_type, mask_type, N_s, head_dim)
        h_sender = h_sender.reshape(B, N_r, num_neighbor, self.num_heads, head_dim).permute(0, 3, 1, 2, 4)
        h_sender = h_sender.reshape(B, self.num_heads // mask_type, mask_type, N_r, num_neighbor, head_dim)
        ## Only receiver need query
        q = self.q_proj(x_state[:, :N_r]).reshape(B, N_r, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(B, self.num_heads // mask_type, mask_type, N_r, head_dim)
        self_attn_std, inter_attn_std = attn_rotation(q, x_receiver, x_sender, h_sender, rs_mask, h_rs_mask, self.scale, eps=self.eps,  **kwargs)
            

        # Split related part
        ## self part
        self_x = torch.matmul(self_attn_std, x_sender) + torch.sum(self_attn_std, dim=-1, keepdim=True) * x_receiver
        self_x = self_x.reshape(B, self.num_heads, N_r, head_dim)
        # Only the garment part is updated
        self_x = self_x.transpose(1, 2).reshape(B, N_r, C)
        orig_rot_rs = x_receiver.unsqueeze(4) + h_sender
        rot_rs = orig_rot_rs[:, :, 0:1]
        h_rotation_hidden = self.rot_mapping(h_rotation, group_input=False)
        h_rotation_hidden = h_rotation_hidden.unsqueeze(2)
        rot_rs = torch.matmul(h_rotation_hidden, rot_rs.unsqueeze(-1)).squeeze(-1)
        assert rot_rs.shape[2] == 1
        rot_rs = self.rs_mlp(rot_rs)
        # Rot each head back to original space
        rot_rs = torch.matmul(h_rotation_hidden.transpose(-1, -2), rot_rs.unsqueeze(-1)).squeeze(-1)
        rot_rs *= inter_attn_std
        rot_rs = rot_rs.reshape(B, self.num_heads//mask_type, N_r, num_neighbor, head_dim)
        rot_rs = rot_rs.permute(0, 2, 3, 1, 4).reshape(B, N_r, num_neighbor, C // mask_type)
        rot_rs *= h_rs_mask.unsqueeze(-1)
        rot_x = torch.sum(rot_rs, dim=-2)

        mesh_x = torch.zeros_like(rot_x).to(rot_x)
        obstacle_x = torch.cat([rot_x, mesh_x], dim=-1)
        x = self_x + obstacle_x
        x = x * self.n_weight.reshape(1,1,-1) + self.n_bias.reshape(1,1,-1)

        x_receiver = x_receiver.reshape(B, self.num_heads, N_r, head_dim).permute(0, 2, 1, 3).reshape(B, N_r, C)
        x_sender = x_sender.reshape(B, self.num_heads, N_s, head_dim).permute(0, 2, 1, 3).reshape(B, N_s, C)
        h_sender = h_sender.reshape(B, self.num_heads, N_r, num_neighbor, head_dim).permute(0, 2, 3, 1, 4).reshape(B, N_r, num_neighbor, C)
        
        x = self.norms[0](x)
        x_receiver = self.norms[1](x_receiver)
        x_sender = self.norms[2](x_sender)
        h_sender = self.norms[2](h_sender)
        x = self.proj(x)
        x_receiver = self.r_proj(x_receiver)
        x_sender = self.s_proj(x_sender)
        h_sender = self.s_proj(h_sender)

        x = torch.cat([x, x_state[:, N_r:]], dim=1)
        rot_weight = self.rot_mapping.get_weight()
        return x, x_receiver, x_sender, h_state, h_sender, h_rotation, rot_weight
