import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, ModuleList
from mmcv.ops import QueryAndGroup, grouping_operation

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from layersnet.utils import face_normals_batched, rotation_from_normals, extract_rotation_with_padding
from layersnet.models.utils import (FFN, RotAttention, RotFFN)
from layersnet.datasets.utils import normalize


class MultiheadAttention(BaseModule):
    def __init__(self, embed_dims, num_heads, dropout=0.0, **kwargs):
        super(MultiheadAttention, self).__init__()
        assert embed_dims % num_heads == 0, 'embed_dims must be ' \
            f'divisible by num_heads. got {embed_dims} and {num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = RotAttention(embed_dims, num_heads, dropout, dense=True, **kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x_state, x_receiver, x_sender,
                h_state, h_sender,
                h_rotation,
                x_rs_mask, h_rs_mask,
                residual, h_residual,
                residual_r, residual_s, residual_hs,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        """
        if residual is None:
            residual = x_state
        if h_residual is None:
            h_residual = h_state
        
        x_state, x_receiver, x_sender, h_state, h_sender, h_rotation, r_weight = self.attn(
            x_state, x_receiver, x_sender,
            h_state, h_sender,
            h_rotation,
            rs_mask=x_rs_mask, h_rs_mask=h_rs_mask,
            # Residual parts
            residual_r=residual_r, residual_s=residual_s, residual_hs=residual_hs, **kwargs)

        return residual + self.dropout(x_state), x_receiver, x_sender, h_residual + self.dropout(h_state), h_sender, h_rotation, r_weight


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in transformer.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(TransformerEncoderLayer, self).__init__()
        assert isinstance(order, tuple) and len(order) == 4
        assert set(order) == set(['selfattn', 'norm', 'ffn'])
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.pre_norm = order[0] == 'norm'
        self.self_attn = MultiheadAttention(embed_dims, num_heads, dropout, act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs)
        self.ffn = FFN([embed_dims, 2 * embed_dims, embed_dims], final_act=True, bias=True, add_residual=True)
        self.norms = ModuleList()
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.res_norms = ModuleList()
        self.res_norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.res_norms.append(build_norm_layer(norm_cfg, embed_dims)[1])

    def forward(self, x_state, x_receiver, x_sender, h_state, h_sender, h_rotation, x_rs_mask, h_rs_mask, **kwargs):
        """Forward function for `TransformerEncoderLayer`.
        """
        norm_cnt = 0
        inp_residual = x_state
        inp_res_r = x_receiver
        inp_res_s = x_sender
        inp_h_residual = h_state
        inp_res_hs = h_sender
        if self.pre_norm:
            # Norm x
            x_state = self.norms[norm_cnt](x_state)
            h_state = self.norms[norm_cnt](h_state)
            norm_cnt += 1
            # Norm receiver and sender
            x_receiver = self.res_norms[0](x_receiver)
            x_sender = self.res_norms[1](x_sender)
            h_sender = self.res_norms[1](h_sender)

        # self attn
        x_state, x_receiver, x_sender, h_state, h_sender, h_rotation, r_weight = self.self_attn(
            x_state, x_receiver, x_sender,
            h_state, h_sender,
            h_rotation,
            x_rs_mask, h_rs_mask,
            # Residual parts
            inp_residual if self.pre_norm else None,
            inp_h_residual if self.pre_norm else None,
            inp_res_r, inp_res_s, inp_res_hs,
            **kwargs)
        inp_residual = x_state
        inp_h_residual = h_state

        # norm
        x_state = self.norms[norm_cnt](x_state)
        h_state = self.norms[norm_cnt](h_state)
        norm_cnt += 1

        # ffn
        x_state = self.ffn(x_state, inp_residual if self.pre_norm else None)
        h_state = self.ffn(h_state, inp_h_residual if self.pre_norm else None)

        # norm
        if not self.pre_norm:
            # Norm x
            x_state = self.norms[norm_cnt](x_state)
            h_state = self.norms[norm_cnt](h_state)
            norm_cnt += 1
            # Norm receiver and sender
            x_receiver = self.res_norms[0](x_receiver)
            x_sender = self.res_norms[1](x_sender)
            h_sender = self.res_norms[1](h_sender)

        return x_state, x_receiver, x_sender, h_state, h_sender, h_rotation, r_weight


class TransformerEncoder(BaseModule):
    """Implements the encoder in transformer.
    """

    def __init__(self,
                 num_layers,
                 embed_dims,
                 num_heads,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'), **kwargs):
        super(TransformerEncoder, self).__init__()
        assert isinstance(order, tuple) and len(order) == 4
        assert set(order) == set(['selfattn', 'norm', 'ffn'])
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.pre_norm = order[0] == 'norm'
        self.layers = ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(embed_dims, num_heads, dropout, order, act_cfg, norm_cfg, **kwargs))
        self.norm = build_norm_layer(norm_cfg,
                                     embed_dims)[1] if self.pre_norm else None

    def forward(self, x_state, x_receiver, x_sender, h_state, h_sender, h_rotation, x_rs_mask, h_rs_mask, **kwargs):
        """Forward function for `TransformerEncoder`.
        """
        rot_weights = []
        # Need follow the exact order to apply the different mask
        for layer in self.layers:
            x_state, x_receiver, x_sender, h_state, h_sender, h_rotation, r_weight = layer(
                x_state, x_receiver, x_sender,
                h_state, h_sender,
                h_rotation,
                x_rs_mask, h_rs_mask, **kwargs)
            rot_weights.append(r_weight)
        if self.norm is not None:
            x_state = self.norm(x_state)
        return x_state, rot_weights


@BACKBONES.register_module()
class LayersNet(BaseBackbone):
    """Implements the LayersNet.
    """

    def __init__(self,
                 attr_dim=5,
                 state_dim=9,
                 normal_dim=3,
                 invisible_forces=5,
                 gravity_dim=3,
                 position_dim=3,
                 embed_dims=128, 
                 num_heads=8,
                 num_encoder_layers=4,
                 dropout=0.0,
                 eps=1e-7,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_fcs=1,
                 group_cfg=None,
                 gh_group_cfg=None,
                 patch_neighbor=4,
                 **kwargs):
        super(LayersNet, self).__init__()
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.normal_dim = normal_dim
        self.invisible_forces_dim = invisible_forces
        self.position_dim = position_dim
        self.embed_dims = embed_dims
        self.eps = eps
        self.gravity_dim = gravity_dim
        self.patch_neighbor = patch_neighbor
        
        # Invisible forces
        self.invisible_force_encoder = FFN(
            [invisible_forces] + [embed_dims for i in range(num_fcs)],
            final_act=True, bias=True)
        self.gravity_encoder = FFN(
            [gravity_dim] + [embed_dims for i in range(num_fcs)],
            final_act=True, bias=True)
        # Attribute
        self.attr_encoder = FFN(
            [attr_dim] + [embed_dims for i in range(num_fcs)],
            final_act=True, bias=True)
        # States embeddings
        self.input_projection = FFN(
            [state_dim] + [embed_dims for i in range(num_fcs)], 
            final_act=True, bias=True)
        ## Receiver and sender
        self.receiver_proj = FFN(
            [embed_dims] + [embed_dims for i in range(num_fcs)], 
            final_act=True, bias=True)
        self.sender_proj = FFN(
            [embed_dims] + [embed_dims for i in range(num_fcs)], 
            final_act=True, bias=True)

        if norm_cfg is not None:
            self.receiver_norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.sender_norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.particle_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.receiver_norm = nn.Identity()
            self.sender_norm = nn.Identity()
            self.particle_norm = nn.Identity()

        assert group_cfg is not None
        self.grouper = QueryAndGroup(**group_cfg)
        self.rot_mapping = RotFFN([position_dim, embed_dims], num_groups=1, final_act=False, bias=False, act_cfg=act_cfg)
        assert gh_group_cfg is not None
        self.gh_group_cfg = gh_group_cfg
        self.gh_grouper = QueryAndGroup(**gh_group_cfg)

        self.encoder = TransformerEncoder(num_encoder_layers, embed_dims,
                                          num_heads,
                                          dropout, order, act_cfg,
                                          norm_cfg, **kwargs)

    def forward(self,
                state,
                h_state, attr, invisable_force, gravity,
                g2p_mask,
                attr_mask,
                invisable_force_mask,
                mean_std,
                radius,
                human_radius,
                h_faces=None,
                h_v2f_mask_sparse=None,
                gmesh_mask=None,
                pmesh_mask=None,
                invisable_rotation=None,
                **kwargs):
        """Forward function.
            
            Order: [garment, wind, gravity] and [human].
        """
        state = state.transpose(-1, -2)
        h_state = h_state.transpose(-1, -2)
        attr = attr.transpose(-1, -2)
        g2p_mask = g2p_mask.transpose(-1, -2).squeeze(1)
        attr_mask = attr_mask.transpose(-1, -2).squeeze(1)
        
        invisable_force_mask = invisable_force_mask.transpose(-1, -2).squeeze(1)
        h_v2f_mask_sparse = torch.sparse_coo_tensor(**h_v2f_mask_sparse)
            
        # Compute patch
        bs, num_garment, num_patch = g2p_mask.shape
        num_human = h_state.shape[1]
        num_invisable_force = invisable_force.shape[1]
        g2p_mask_T = g2p_mask.transpose(-1, -2)

        # Calculate neighbors before normalizing inputs
        patch_state_raw = torch.bmm(g2p_mask_T, state) / (torch.sum(g2p_mask_T, dim=-1, keepdim=True) + self.eps)
        ## human states are seperated
        sender_state = patch_state_raw
        sender_pos = sender_state[:, :, :self.position_dim]
        reciever_pos = patch_state_raw[:, :, :self.position_dim]
        rs_mask = torch.sqrt(torch.sum(reciever_pos**2, dim=-1, keepdim=True) + torch.sum(sender_pos**2, dim=-1, keepdim=True).transpose(-1, -2) - 2 * torch.bmm(reciever_pos, sender_pos.transpose(-1, -2)))
        # Find neighbor patches
        rs_mask = rs_mask < radius.unsqueeze(1).unsqueeze(1)
        # rs_mask = rs_mask.bool()
        padding_mask = torch.sum(g2p_mask_T, dim=-1, keepdim=True) > 0
        rs_mask *= padding_mask.transpose(-1, -2)
        rs_mask *= padding_mask
        p2a_mask = torch.bmm(g2p_mask_T, attr_mask) > 0
        p2i_mask = torch.bmm(g2p_mask_T, invisable_force_mask) > 0
        rs_mask = torch.cat([rs_mask, p2a_mask], dim=-1)
        rs_mask.requires_grad = False
        rs_mask = rs_mask.unsqueeze(1)
        if pmesh_mask is not None:
            pmesh_mask = pmesh_mask.transpose(-1, -2).squeeze(1)
        else:
            assert gmesh_mask is not None
            gmesh_mask = gmesh_mask.transpose(-1, -2).squeeze(1)
            pmesh_mask = torch.bmm(torch.bmm(g2p_mask_T, gmesh_mask), g2p_mask)
        pmesh_mask = torch.cat(
            [pmesh_mask, p2a_mask], dim=-1).bool()

        pmesh_mask = pmesh_mask.unsqueeze(1)
        # Clean the mask in rs_mask that already in pmesh_mask
        rs_mask *= ~pmesh_mask
        rs_mask = torch.cat([rs_mask, pmesh_mask], dim=1)

        # Find neighbor human vertices
        ph_group_xyz_diff, ph_group_idx = self.gh_grouper(h_state[:, :, :self.position_dim].contiguous(), patch_state_raw[:, :, :self.position_dim].contiguous())
        ph_group_xyz_diff = ph_group_xyz_diff.permute(0, 2, 3, 1)
        ph_group_xyz_l2 = torch.sqrt(torch.sum(ph_group_xyz_diff**2, dim=-1, keepdim=True))
        ph_group_xyz_l2_mask = ph_group_xyz_l2 < human_radius.unsqueeze(1).unsqueeze(1)
        ph_group_xyz_l2_mask.requires_grad = False
        h_rs_mask = ph_group_xyz_l2_mask.squeeze(-1)
        ## Find nearest for further useage
        patch_nn_human = self.grouper(h_state[:, :, :self.position_dim].contiguous(), patch_state_raw[:, :, :self.position_dim].contiguous())

        particle_nn_human = self.grouper(h_state[:, :, :self.position_dim].contiguous(), state[:, :, :self.position_dim].contiguous())

        human_face_normals = face_normals_batched(h_state[..., :3], h_faces.transpose(-1, -2))
        human_vert_normals = torch.bmm(h_v2f_mask_sparse, human_face_normals)
        human_vert_normals = human_vert_normals / (torch.linalg.norm(human_vert_normals, dim=-1, ord=2, keepdim=True) + self.eps)
        human_vert_rotations = rotation_from_normals(human_vert_normals)
        h_rotation = grouping_operation(human_vert_rotations.reshape(bs, num_human, -1).transpose(-1, -2), ph_group_idx).permute(0, 2, 3, 1).reshape(bs, num_patch, -1, 3, 3)

        gravity_rotations = rotation_from_normals(gravity[:, :, -3:])
            
        # Normalize for inputs
        state_normed, h_state_normed = normalize([state, h_state], [mean_std, mean_std])
        patch_state_normed = torch.bmm(g2p_mask_T, state_normed) / (torch.sum(g2p_mask_T, dim=-1, keepdim=True) + self.eps)

        # Common vectors
        # States
        patch_state_X = self.input_projection(patch_state_normed)
        # Human
        h_state_X = self.input_projection(h_state_normed)
        patch_state_input = patch_state_X
        h_state_input = h_state_X
        # Get corresponding human information
        patch_corresponding_human = grouping_operation(h_state_input.transpose(-1, -2), ph_group_idx).permute(0, 2, 3, 1)
        patch_corresponding_human *= ph_group_xyz_l2_mask
        # Attribute 
        attr = self.attr_encoder(attr)
        # Invisible forces
        ## To combine with human
        invisable_force = self.invisible_force_encoder(invisable_force)
        gravity = self.gravity_encoder(gravity)

        # Combine forces with human
        patch_corresponding_human = torch.cat([
            patch_corresponding_human,
            invisable_force.unsqueeze(1).expand(-1, num_patch, -1, -1),
            gravity.unsqueeze(1).expand(-1, num_patch, -1, -1)], dim=2)
        h_rs_mask = torch.cat([h_rs_mask, p2i_mask, p2i_mask], dim=-1)
        h_rotation = torch.cat([
            h_rotation,
            invisable_rotation[:, -1:].unsqueeze(1).expand(-1, num_patch, -1, -1, -1),
            gravity_rotations.unsqueeze(1).expand(-1, num_patch, -1, -1, -1)], dim=2)
        # Stage 1
        ## Patch learning
        ## Need follow the exact order to apply the different mask
        patch_x_state = torch.cat([patch_state_input, attr], dim=1)
        patch_x_receiver = self.receiver_proj(patch_x_state[:, :num_patch])
        patch_x_sender = self.sender_proj(patch_x_state)
        h_sender = self.sender_proj(patch_corresponding_human)
        # Norm
        patch_x_state = self.particle_norm(patch_x_state)
        patch_x_receiver = self.receiver_norm(patch_x_receiver)
        patch_x_sender = self.sender_norm(patch_x_sender)
        h_sender = self.sender_norm(h_sender)

        patch_emb, rot_weights = self.encoder(
            patch_x_state, patch_x_receiver, patch_x_sender,
            patch_corresponding_human, h_sender,
            h_rotation,
            rs_mask, h_rs_mask)

        # Stage 2
        # Find candidate patches
        unique_g2p_index = torch.argmax(g2p_mask, dim=-1, keepdim=True).int()
        g2p_neighbor = grouping_operation(pmesh_mask[..., :num_patch].squeeze(1).transpose(-1, -2).float(), unique_g2p_index).permute(0, 2, 3, 1).squeeze(-2)
        num_neighbor = torch.max(torch.sum(g2p_neighbor, dim=-1)).int().item()
        p_neighbor_mask, p_neighbor_index = torch.topk(
            g2p_neighbor, 
            num_neighbor, dim=-1)
        particle_corresponding_patch_raw = grouping_operation(patch_state_raw.transpose(-1, -2), unique_g2p_index).permute(0, 2, 3, 1)
        ## This is for relative prediction
        rel_state_raw = state - particle_corresponding_patch_raw.squeeze(-2)
        particle_corresponding_patch_neighbor = grouping_operation(patch_state_raw.transpose(-1, -2), p_neighbor_index.int()).permute(0, 2, 3, 1)
        # Filter too far away patches
        patch_distant = torch.sqrt(torch.sum(
            (particle_corresponding_patch_neighbor[..., :self.position_dim]-particle_corresponding_patch_raw[..., :self.position_dim])**2,
            dim=-1))
        particle_distant = torch.sqrt(torch.sum(
            (particle_corresponding_patch_neighbor[..., :self.position_dim]-state[..., :self.position_dim].unsqueeze(-2))**2,
            dim=-1))
        belong_patch = torch.where(patch_distant == 0)
        distant_mask = particle_distant < patch_distant
        distant_mask[belong_patch] = True
        valid_mask = distant_mask & p_neighbor_mask.bool()

        # Filter for the second time, only topk to save more space
        selected_n_val, selected_n_idx = torch.topk(particle_distant, self.patch_neighbor, dim=-1, largest=False)
        selected_p_neighbor_index = torch.gather(p_neighbor_index, -1, selected_n_idx)
        selected_valid_mask = torch.gather(valid_mask, -1, selected_n_idx)
        selected_cor_patch_raw = grouping_operation(patch_state_raw.transpose(-1, -2), selected_p_neighbor_index.int()).permute(0, 2, 3, 1)
        rel_state = state.unsqueeze(-2) - selected_cor_patch_raw
        particle_corresponding_p_neighbor_emb = grouping_operation(patch_emb.transpose(-1, -2), selected_p_neighbor_index.int()).permute(0, 2, 3, 1)

        group_xyz_diff, group_idx = particle_nn_human
        particle_corresponding_human = grouping_operation(h_state_input.transpose(-1, -2), group_idx).permute(0, 3, 2, 1).squeeze(1)
        group_xyz_diff = group_xyz_diff.permute(0, 3, 2, 1).squeeze(1)
        group_xyz_l2 = torch.sqrt(torch.sum(group_xyz_diff**2, dim=-1, keepdim=True))
        group_xyz_l2_mask = group_xyz_l2 < human_radius.unsqueeze(1).unsqueeze(1)
        group_xyz_l2_mask.requires_grad = False

        # Get corresponding human, if too far then mask as 0s
        particle_corresponding_human *= group_xyz_l2_mask

        # Rotation in hidden states
        decode_rotation = extract_rotation_with_padding(human_vert_rotations, group_idx, mask=group_xyz_l2_mask, pad_dim=self.position_dim)
        h_rotation_hidden = self.rot_mapping(decode_rotation, group_input=False)[:, 0]

        # Rotate decoder input seperately
        rot_patch_state = torch.matmul(h_rotation_hidden, particle_corresponding_p_neighbor_emb.transpose(-1, -2)).transpose(-1, -2)
        rot_human = torch.matmul(h_rotation_hidden, particle_corresponding_human.unsqueeze(-1)).squeeze(-1)
        rel_state_rot = torch.matmul(decode_rotation, rel_state.reshape(bs, num_garment, -1, 3).transpose(-1, -2))
        rel_state_rot = rel_state_rot.transpose(-1, -2).reshape(bs, num_garment, self.patch_neighbor, -1, 3).reshape(bs, num_garment, self.patch_neighbor, -1)
        garment_enc = torch.cat([rel_state_rot, rot_patch_state, rot_human.unsqueeze(-2).expand(-1, -1, self.patch_neighbor, -1)], dim=-1)

        rot_weights.append(self.rot_mapping.get_weight())

        # For patch dynamic predictions
        ## Rotate
        p_group_xyz_diff, p_group_idx = patch_nn_human
        p_group_xyz_diff = p_group_xyz_diff.permute(0, 2, 3, 1).squeeze(-2)
        p_group_xyz_l2 = torch.sqrt(torch.sum(p_group_xyz_diff**2, dim=-1, keepdim=True))
        p_group_xyz_l2_mask = p_group_xyz_l2 < human_radius.unsqueeze(1).unsqueeze(1)
        p_group_xyz_l2_mask.requires_grad = False
        patch_decode_rotation = extract_rotation_with_padding(human_vert_rotations, p_group_idx, mask=p_group_xyz_l2_mask, pad_dim=self.position_dim)
        ph_rotation_hidden = self.rot_mapping(patch_decode_rotation, group_input=False)[:, 0]
        rot_patch_emb = torch.matmul(ph_rotation_hidden, patch_emb[:, :num_patch].unsqueeze(-1)).squeeze(-1)
        patch_corresponding_human = grouping_operation(h_state_input.transpose(-1, -2), p_group_idx).permute(0, 2, 3, 1).squeeze(-2)
        rot_human = torch.matmul(ph_rotation_hidden, patch_corresponding_human.unsqueeze(-1)).squeeze(-1)
        patch_enc = torch.cat([torch.zeros_like(patch_state_raw).to(patch_state_raw), rot_patch_emb, rot_human], dim=-1)

        return dict(
            feature=garment_enc,
            rot_weights=rot_weights, decode_rotation=decode_rotation,
            patch_feature=patch_enc, patch_decode_rotation=patch_decode_rotation,
            g2p_index=unique_g2p_index,
            p_state=patch_state_raw.transpose(-1, -2),
            g2p_rel_state=rel_state_raw.transpose(-1, -2),
            patch_mask=selected_valid_mask,
            patch_Z=None)
