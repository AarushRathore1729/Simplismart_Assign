import torch
from transformers.modeling_outputs import BaseModelOutput
from .nucleus_sampler import top_p_sample

def speculative_decode_greedy(target_model, draft_model, target_enc, draft_enc, prefix, max_new_tokens, draft_k, eos_token_id):
    tokens = prefix.clone()
    target_enc_out = BaseModelOutput(last_hidden_state=target_enc.last_hidden_state)
    draft_enc_out = BaseModelOutput(last_hidden_state=draft_enc.last_hidden_state)
    
    while tokens.shape[1] < prefix.shape[1] + max_new_tokens:
        draft_tokens, draft_cache = tokens.clone(), None
        for _ in range(draft_k):
            input_ids = draft_tokens[:, -1:] if draft_cache else draft_tokens
            out = draft_model(decoder_input_ids=input_ids, encoder_outputs=draft_enc_out, past_key_values=draft_cache, use_cache=True)
            draft_cache = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens = torch.cat([draft_tokens, next_tok], dim=1)
            if next_tok.item() == eos_token_id: break
        
        proposed = draft_tokens[:, tokens.shape[1]:]
        if proposed.shape[1] == 0: break
        
        target_out = target_model(decoder_input_ids=draft_tokens, encoder_outputs=target_enc_out, use_cache=False)
        target_preds = torch.argmax(target_out.logits[:, tokens.shape[1]-1:-1, :], dim=-1)
        
        matches = (proposed == target_preds)
        if matches.all(): tokens = draft_tokens
        else:
            mismatch_idx = (~matches).nonzero(as_tuple=True)[1][0].item()
            tokens = torch.cat([tokens, proposed[:, :mismatch_idx], target_preds[:, mismatch_idx:mismatch_idx+1]], dim=1)
        
        if tokens[0, -1].item() == eos_token_id: break
    return tokens

def speculative_decode_top_p(target_model, draft_model, target_enc, draft_enc, prefix, max_new_tokens, draft_k, eos_token_id, top_p, temperature=1.0):
    tokens = prefix.clone()
    target_enc_out = BaseModelOutput(last_hidden_state=target_enc.last_hidden_state)
    draft_enc_out = BaseModelOutput(last_hidden_state=draft_enc.last_hidden_state)
    
    while tokens.shape[1] < prefix.shape[1] + max_new_tokens:
        draft_tokens, draft_probs_list, draft_cache = tokens.clone(), [], None
        for _ in range(draft_k):
            input_ids = draft_tokens[:, -1:] if draft_cache else draft_tokens
            out = draft_model(decoder_input_ids=input_ids, encoder_outputs=draft_enc_out, past_key_values=draft_cache, use_cache=True)
            draft_cache = out.past_key_values
            logits = out.logits[:, -1, :]
            next_tok = top_p_sample(logits, top_p, temperature)
            probs = torch.softmax(logits / temperature if temperature != 1.0 else logits, dim=-1)
            draft_probs_list.append(probs.gather(-1, next_tok))
            draft_tokens = torch.cat([draft_tokens, next_tok], dim=1)
            if next_tok.item() == eos_token_id: break
        
        proposed = draft_tokens[:, tokens.shape[1]:]
        if proposed.shape[1] == 0: break
        draft_probs = torch.cat(draft_probs_list, dim=1)
        
        target_out = target_model(decoder_input_ids=draft_tokens, encoder_outputs=target_enc_out, use_cache=False)
        target_logits = target_out.logits[:, tokens.shape[1]-1:-1, :]
        if temperature != 1.0: target_logits = target_logits / temperature
        target_probs = torch.softmax(target_logits, dim=-1).gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
        
        accepted = torch.rand_like(target_probs) < torch.clamp(target_probs / (draft_probs + 1e-10), max=1.0)
        if accepted.all():
            tokens = torch.cat([draft_tokens, top_p_sample(target_out.logits[:, -1, :], top_p, temperature)], dim=1)
        else:
            reject_idx = (~accepted).nonzero(as_tuple=True)[1][0].item()
            tokens = torch.cat([tokens, proposed[:, :reject_idx], top_p_sample(target_logits[:, reject_idx, :], top_p, temperature)], dim=1)
        
        if tokens[0, -1].item() == eos_token_id: break
    return tokens

def speculative_decode(target_model, draft_model, target_enc, draft_enc, prefix, max_new_tokens, draft_k, eos_token_id, top_p=0.0, temperature=1.0):
    if top_p == 0.0: return speculative_decode_greedy(target_model, draft_model, target_enc, draft_enc, prefix, max_new_tokens, draft_k, eos_token_id)
    return speculative_decode_top_p(target_model, draft_model, target_enc, draft_enc, prefix, max_new_tokens, draft_k, eos_token_id, top_p, temperature)
