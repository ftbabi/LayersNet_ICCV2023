# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from .. import builder
from ..builder import SIMULATORS
from .base import BaseSimulator
from layersnet.core import add_prefix
from layersnet.datasets.utils import to_numpy_detach


@SIMULATORS.register_module()
class EncoderDecoder(BaseSimulator):
    def __init__(self,
                 backbone,
                 decode_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and simulator set pretrained weight'
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = builder.build_backbone(backbone)
        self._init_decode_head(decode_head)

        # This is for preprocess/augment train input
        self.train_cfg = train_cfg
        # This is for preprocess/augment test input
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)

    def init_weights(self):
        super(EncoderDecoder, self).init_weights()
            
    def extract_feat(self, inputs, **kwargs):
        """Extract features from inputs."""
        x = self.backbone(**inputs, **kwargs)
        return x

    def encode_decode(self, inputs, gt_label=None, **kwargs):
        x = self.extract_feat(inputs, is_test=True)
        out = self._decode_head_forward_test(x, inputs, gt_label=gt_label)
        return out

    def _decode_head_forward_train(self, x, inputs, gt_label):
        label = gt_label['vertices'] if gt_label is not None else None
        losses = dict()
        loss_decode, pred = self.decode_head.forward_train(**x, **inputs,
                                                     gt_label=label,
                                                     train_cfg=self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, pred

    def _decode_head_forward_test(self, x, inputs, gt_label=None):
        label = gt_label['vertices'] if gt_label is not None else None
        logits = self.decode_head.forward_test(**x, **inputs, gt_label=label, test_cfg=self.test_cfg)
        return logits

    def forward_train(self, inputs, gt_label, **kwargs):
        if self.train_cfg is not None and self.train_cfg.get('non_grad_step', 0) > 0:
            assert self.train_cfg.get('step', 1) == 1
            assert len(inputs['h_state'].shape) > 3
            frame_state_dim = gt_label['vertices'].shape[1]
            state_list = inputs.pop('state', None)
            h_state_list = inputs.pop('h_state', None)
            invisible_force_list = inputs.pop('invisable_force', None)
            invisable_rotation_list = inputs.pop('invisable_rotation', None)
            gravity_list = inputs.pop('gravity', None)
            non_grad_step = h_state_list.shape[1] - 1
            bs = h_state_list.shape[0]
            assert h_state_list is not None and h_state_list.shape[1] == non_grad_step + 1
            # Generate randomly mask sequence
            seq_selected = torch.randint(0, non_grad_step+1, (bs, 1, 1)).to(state_list)
            self.eval()
            with torch.no_grad():
                inputs['state'] = state_list[:, 0]
                inputs['h_state'] = h_state_list[:, 0]
                inputs['invisable_force'] = invisible_force_list[:, 0]
                if invisable_rotation_list is not None:
                    inputs['invisable_rotation'] = invisable_rotation_list[:, 0]
                if gravity_list is not None:
                    inputs['gravity'] = gravity_list[:, 0]
                step_inputs = inputs
                for ng_step in range(non_grad_step):
                    pred = self.inference(step_inputs)
                    pred_state = pred['pred'][:, :, :frame_state_dim].transpose(-1, -2)
                    # Move the history
                    seq_mask = ng_step >= seq_selected
                    current_state = seq_mask * pred_state + (~seq_mask) * state_list[:, ng_step+1][:, :frame_state_dim]
                    step_inputs['state'] = torch.cat([current_state, step_inputs['state'][:, :-frame_state_dim]], dim=1)
                    step_inputs['h_state'] = h_state_list[:, ng_step+1]
                    step_inputs['invisable_force'] = invisible_force_list[:, ng_step+1]
                    if invisable_rotation_list is not None:
                        step_inputs['invisable_rotation'] = invisable_rotation_list[:, ng_step+1]
                    if gravity_list is not None:
                        step_inputs['gravity'] = gravity_list[:, ng_step+1]

            self.train()
            inputs = step_inputs

        x = self.extract_feat(inputs)
        losses = dict()
        loss_decode, pred = self._decode_head_forward_train(x, inputs=inputs, gt_label=gt_label)
        losses.update(loss_decode)

        return losses

    def inference(self, inputs, **kwargs):
        output = self.encode_decode(inputs, **kwargs)
        return output

    def evaluate(self, pred, gt_label, meta_info, **kwargs):
        pass

    def simple_test(self, inputs, **kwargs):
        pred = self.inference(inputs, **kwargs)
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = to_numpy_detach(pred)
        return pred
