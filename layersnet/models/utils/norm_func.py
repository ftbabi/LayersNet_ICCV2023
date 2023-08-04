import torch


def attn_rotation(query, receiver_emb, sender_emb, inter_sender_emb, self_attn_mask, inter_attn_mask, scale, eps=1e-7, **kwargs):
    bs, _num_head, _mask_type, q_N, head_dim = query.shape
    N_r = receiver_emb.shape[3]
    N_s = sender_emb.shape[3]
    N_nn = inter_sender_emb.shape[4]

    self_attn_mask = self_attn_mask.unsqueeze(1)
    inter_attn_mask = inter_attn_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)

    # Calculate self attention weight
    r_square = torch.sum(receiver_emb * receiver_emb, dim=-1, keepdim=True)
    s_square = torch.sum(sender_emb * sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
    rs = torch.matmul(receiver_emb, sender_emb.transpose(-1, -2))
    self_var_rs = (r_square + s_square-2*rs)/4
    self_std_rs = torch.sqrt(self_var_rs)
    self_attn = torch.matmul(query, sender_emb.transpose(-1, -2)) + torch.sum(query * receiver_emb, dim=-1, keepdim=True)
    self_attn /= (self_std_rs+eps)
    self_attn = self_attn * scale
    self_attn.masked_fill_(~self_attn_mask, float('-inf'))

    # Calculate inter attention weight
    ### mask_type 0: world mask; 1: mesh mask
    inter_query = query[:, :, 0:1].unsqueeze(4)
    inter_receiver = receiver_emb[:, :, 0:1].unsqueeze(4)
    inter_sender = inter_sender_emb[:, :, 0:1]
    inter_var_rs = torch.sum((inter_receiver - inter_sender)**2, dim=-1, keepdim=True) / 4
    inter_std_rs = torch.sqrt(inter_var_rs)
    raw_inter_attn = torch.sum(inter_query*inter_sender, dim=-1, keepdim=True) + torch.sum(inter_query*inter_receiver, dim=-1, keepdim=True)
    inter_attn =raw_inter_attn / (inter_std_rs+eps)
    inter_attn = inter_attn * scale
    inter_attn.masked_fill_(~inter_attn_mask, float('-inf'))
    inter_attn = inter_attn.squeeze(-1)
    
    # Joint softmax
    ### mask_type 0: world mask; 1: mesh mask
    joint_attn = torch.cat([
        self_attn,
        torch.cat([inter_attn, torch.full_like(inter_attn, float('-inf')).to(inter_attn)], dim=2)
    ], dim=-1)
    joint_attn = joint_attn.softmax(dim=-1)

    # Split attn
    self_attn_normed = joint_attn[..., :N_s]
    inter_attn_normed = joint_attn[:, :, 0:1, :, N_s:].unsqueeze(-1)

    # Clean NaN
    self_attn_normed = self_attn_normed.masked_fill(~self_attn_mask, 0.0).clamp(-5, 5)
    inter_attn_normed = inter_attn_normed.masked_fill(~inter_attn_mask, 0.0).clamp(-5, 5)
    assert torch.where(torch.isnan(self_attn_normed))[0].size(0) == 0
    assert torch.where(torch.isnan(inter_attn_normed))[0].size(0) == 0

    # For aggregation
    self_attn_normed = self_attn_normed / (self_std_rs * 2)
    inter_attn_normed = inter_attn_normed / (inter_std_rs * 2)
    return self_attn_normed, inter_attn_normed