import torch

def top_p_sample(logits, top_p, temperature=1.0):
    if temperature != 1.0: logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.gather(sorted_indices, dim=-1, index=torch.multinomial(sorted_probs, num_samples=1))
