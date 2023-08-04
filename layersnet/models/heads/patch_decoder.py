import torch
from mmcv.cnn import build_activation_layer
from mmcv.ops import QueryAndGroup, grouping_operation

from ..builder import HEADS
from .sim_head import SimHead
from layersnet.models.utils import FFN
from layersnet.datasets.utils import denormalize
from layersnet.models.losses import L2Loss


@HEADS.register_module()
class PatchDecoder(SimHead):
    def __init__(self,
                 out_channels=3,
                 in_channels=128*2,
                 dt=1/30,
                 add_residual=True,
                 init_cfg=None,
                 eps=1e-7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 regularizer_weight=1.0,
                 patch_weight=1.0,
                 *args,
                 **kwargs):
        super(PatchDecoder, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        # position_dim
        self.out_channels = out_channels
        self.dt = dt
        self.eps = eps
        self.loss_dim = (0, 3)
        self.regularizer_weight = regularizer_weight
        self.patch_weight = patch_weight

        if self.out_channels <= 0:
            raise ValueError(
                f'num_classes={out_channels} must be a positive integer')

        self.proj = FFN([in_channels, in_channels, in_channels, out_channels], act_cfg=act_cfg, add_residual=(add_residual & (in_channels==out_channels)))
        self.activate = build_activation_layer(dict(type='Sigmoid'))
        # For test
        self.grouper = QueryAndGroup(
            None,
            1,
            min_radius=0.0,
            use_xyz=False,
            normalize_xyz=False,
            return_grouped_xyz=True,
            return_grouped_idx=False,
            return_unique_cnt=False,
        )
        self.orth_regularizer = L2Loss(reduction='sum', loss_weight=regularizer_weight, loss_name='loss_l2_orth')
        self.patch_dynamic_l2 = L2Loss(reduction='sum', loss_weight=patch_weight, loss_name='loss_l2_patch', obj_wise=False)

    def init_weights(self):
        super(PatchDecoder, self).init_weights()
        pass
    
    def evaluate(self, pred, gt_label, **kwargs):
        gt_label = gt_label.transpose(-1, -2)
        h_state = kwargs.pop('h_state', None)
        assert h_state is not None
        h_state = h_state.transpose(-1, -2)[..., self.loss_dim[0]:self.loss_dim[1]]
        acc_dict = self.accuracy(pred[..., self.loss_dim[0]:self.loss_dim[1]], gt_label[..., self.loss_dim[0]:self.loss_dim[1]], h_state=h_state, **kwargs)
        return acc_dict

    def simple_test(self,
            feature, state, gt_label=None, test_cfg=None, h_state=None, h_faces=None, h_v2f_mask_sparse=None,
            decode_rotation=None,
            patch_decode_rotation=None, patch_feature=None, p_state=None, g2p_index=None, g2p_rel_state=None, **kwargs):
        # Patch global xyz
        pred_patch = self.predict(patch_feature, p_state, decode_rotation=patch_decode_rotation, **kwargs)
        pred_g_p = grouping_operation(pred_patch.transpose(-1, -2), g2p_index).permute(0, 2, 3, 1).squeeze(-2)
        pred = self.predict(feature, g2p_rel_state, decode_rotation=decode_rotation, **kwargs)
        pred = pred + pred_g_p

        rst = dict(pred=pred)
        if gt_label is not None:
            acc_dict = self.evaluate(pred, gt_label, reduction_override='none', h_state=h_state, h_faces=h_faces, h_v2f_mask_sparse=h_v2f_mask_sparse, **kwargs) 
            rst.update(dict(acc=acc_dict))
        return rst

    def forward_train(self,
            feature, gt_label, state, rot_weights=[],
            g2p_mask=None,
            decode_rotation=None, patch_decode_rotation=None,
            patch_feature=None, p_state=None, g2p_index=None, g2p_rel_state=None, **kwargs):
        gt_label = gt_label.transpose(-1, -2)

        pred_patch = self.predict(patch_feature, p_state, decode_rotation=patch_decode_rotation, **kwargs)
        pred_g_p = grouping_operation(pred_patch.transpose(-1, -2), g2p_index).permute(0, 2, 3, 1).squeeze(-2)
        pred = self.predict(feature, g2p_rel_state, decode_rotation=decode_rotation, **kwargs)
        pred = pred + pred_g_p
        # Label is not normalized
        losses = self.loss(pred[:, :, self.loss_dim[0]:self.loss_dim[1]], gt_label[:, :, self.loss_dim[0]:self.loss_dim[1]], **kwargs)

        # L2 regularizer for orthogonal matrix
        orth_regularizer_list = []
        for r_weight in rot_weights:
            for r_w in r_weight:
                aaT = torch.mm(r_w, r_w.transpose(-1, -2))
                aTa = torch.mm(r_w.transpose(-1, -2), r_w)
                l_aaT = self.orth_regularizer(aaT.unsqueeze(-1), torch.eye(aaT.shape[0]).unsqueeze(-1).to(aaT))
                l_aTa = self.orth_regularizer(aTa.unsqueeze(-1), torch.eye(aTa.shape[0]).unsqueeze(-1).to(aaT))
                orth_regularizer_list.append(l_aaT+l_aTa)
        if len(orth_regularizer_list) > 0:
            orth_loss = torch.sum(torch.stack(orth_regularizer_list))
            losses[f'{self.orth_regularizer.loss_name}'] = orth_loss

        # L2 for patch_wise dynamics.
        g2p_mask_T = g2p_mask.squeeze(1)
        patch_indices_weight = torch.sum(g2p_mask_T, dim=-1)
        patch_state_raw = torch.bmm(g2p_mask_T, gt_label) / (torch.sum(g2p_mask_T, dim=-1, keepdim=True) + self.eps)
        patch_loss = self.patch_dynamic_l2(
            patch_state_raw[:, :, self.loss_dim[0]:self.loss_dim[1]],
            pred_patch[:, :, self.loss_dim[0]:self.loss_dim[1]],
            indices_weight=patch_indices_weight)
        losses[f'{self.patch_dynamic_l2.loss_name}'] = patch_loss

        return losses, pred

    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)

    def predict(self, garment_emb, state, decode_rotation, mean_std, patch_mask=None, **kwargs):
        state = state.transpose(-1, -2)
        pred_acc = self.proj(garment_emb)
        if len(pred_acc.shape) > 3:
            assert patch_mask is not None
            # Aggregate acc here
            pred_acc = torch.sum(pred_acc * patch_mask.unsqueeze(-1), dim=-2) / (torch.sum(patch_mask, dim=-1, keepdim=True) + self.eps)
        pred_acc = torch.matmul(decode_rotation.transpose(-1, -2), pred_acc.unsqueeze(-1)).squeeze(-1)
        pred_acc = denormalize([pred_acc], [mean_std[:, :, -3:]])[0]
        pred_vel = self.dt * pred_acc + state[:, :, 3:6]
        pred_pos = self.dt * pred_vel + state[:, :, 0:3]
        
        pred = torch.cat([pred_pos, pred_vel, pred_acc], dim=-1)
        return pred