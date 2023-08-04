
def count_parameters(model):
    total_params = 0
    for p, named_p in zip(model.parameters(), model.named_parameters()):
        if p.requires_grad:
            params = p.numel()
            total_params += params
    return total_params