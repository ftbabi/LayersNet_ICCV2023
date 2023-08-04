import torch
import torch.nn as nn
from mmcv.cnn import (Linear, build_activation_layer)

from mmcv.runner import BaseModule


class RotFFN(BaseModule):
    """Implements feed-forward networks (FFNs) with residual connection.
    Args:
        embed_dims (list): The feature dimensions.
        act_cfg (dict, optional): The activation config for FFNs.
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0.0.
        final_act (bool, optional): Add activation after the final layer.
            Defaults to False.
        add_residual (bool, optional): Add resudual connection.
            Defaults to False.
    """

    def __init__(self,
                 embed_dims,
                 num_groups=1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 final_act=False,
                 add_residual=False,
                 bias=False,
                 init_cfg=None,):
        super(RotFFN, self).__init__(init_cfg)
        
        assert isinstance(embed_dims, list)

        self.embed_dims = embed_dims
        self.num_groups = num_groups

        self.act_cfg = act_cfg
        self.dropout = dropout
        if act_cfg is not None:
            self.activate = build_activation_layer(act_cfg)
        else:
            self.activate = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.final_act = final_act
        self.add_residual = add_residual

        # To move the model to device
        self.group_layers = nn.ModuleList()
        for i in range(self.num_groups):
            if len(embed_dims) <= 1:
                layers = nn.Identity()
            else:
                layers = nn.ModuleList()
                in_channels = embed_dims[0]
                out_channels = embed_dims[1]
                for i in range(1, len(embed_dims)-1):
                    layers.append(
                        nn.Sequential(
                            RotLayer(in_channels, out_channels, bias=bias), self.activate,
                            nn.Dropout(dropout)))
                    in_channels = embed_dims[i]
                    out_channels = embed_dims[i+1]
                layers.append(RotLayer(in_channels, out_channels, bias=bias))
                if self.final_act:
                    layers.append(self.activate)
                layers = nn.Sequential(*layers)
            self.group_layers.append(layers)

    def forward(self, x, group_input=False, residual=None):
        """Forward function for `FFN`.
            x: bs, num_verts, dim, dim
        """
        assert not group_input or group_input and x.shape[1] == self.num_groups
        out = []
        for i in range(self.num_groups):
            out_i = self.group_layers[i](x)
            out.append(out_i)
        out = torch.stack(out, dim=1)
        
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

    def get_weight(self):
        weights = []
        for group_layer in self.group_layers:
            gl_weight = group_layer[0].model.weight
            weights.append(gl_weight)
        return weights

class RotLayer(BaseModule):
    """Implements feed-forward networks (FFNs) with residual connection.
    Args:
        embed_dims (list): The feature dimensions.
        act_cfg (dict, optional): The activation config for FFNs.
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0.0.
        final_act (bool, optional): Add activation after the final layer.
            Defaults to False.
        add_residual (bool, optional): Add resudual connection.
            Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 init_cfg=None,
                 **kwargs):
        super(RotLayer, self).__init__(init_cfg)
        self.model = Linear(in_channels, out_channels, bias=bias, **kwargs)
        nn.init.orthogonal_(self.model.weight)

    def forward(self, x):
        """Forward function for `FFN`.
        """
        out = self.model(x.transpose(-1, -2))
        out = self.model(out.transpose(-1, -2))
        return out