import torch
import torch.nn as nn
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      xavier_init)

from mmcv.runner import BaseModule


class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with residual connection.
    """

    def __init__(self,
                 embed_dims,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 final_act=False,
                 add_residual=False,
                 bias=True,
                 init_cfg=None,):
        super(FFN, self).__init__(init_cfg)
        
        assert isinstance(embed_dims, list)

        self.embed_dims = embed_dims

        self.act_cfg = act_cfg
        self.dropout = dropout
        if act_cfg is not None:
            self.activate = build_activation_layer(act_cfg)
        else:
            self.activate = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.final_act = final_act
        self.add_residual = add_residual

        if len(embed_dims) <= 1:
            self.layers = nn.Identity()
        else:
            layers = nn.ModuleList()
            in_channels = embed_dims[0]
            out_channels = embed_dims[1]
            for i in range(1, len(embed_dims)-1):
                layers.append(
                    nn.Sequential(
                        Linear(in_channels, out_channels, bias=bias), self.activate,
                        nn.Dropout(dropout)))
                in_channels = embed_dims[i]
                out_channels = embed_dims[i+1]
            layers.append(Linear(in_channels, out_channels, bias=bias))
            if self.final_act:
                layers.append(self.activate)
            self.layers = nn.Sequential(*layers)

    def forward(self, x, residual=None, **kwargs):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'final_act={self.final_act})'
        repr_str += f'add_residual={self.add_residual})'
        return repr_str

class GroupFFN(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_groups=1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 final_act=False,
                 add_residual=False,
                 bias=True,
                 init_cfg=None,):
        super(GroupFFN, self).__init__(init_cfg)
        
        assert isinstance(embed_dims, list)

        self.embed_dims = embed_dims
        self.num_groups = num_groups

        self.group_layers = nn.ModuleList()
        for i in range(self.num_groups):
            layers = FFN(embed_dims, act_cfg=act_cfg, dropout=dropout, final_act=final_act, add_residual=add_residual, bias=bias, init_cfg=init_cfg)
            self.group_layers.append(layers)

    def forward(self, x, residual=None, **kwargs):
        """Forward function for `FFN`.
        
        """
        assert x.shape[1] == self.num_groups
        out = []
        for i in range(self.num_groups):
            out_i = self.group_layers[i](x[:, i], residual=residual)
            out.append(out_i)
        out = torch.stack(out, dim=1)
        return out
