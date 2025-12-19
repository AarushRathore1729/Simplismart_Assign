import time
import torch
from .decoder import speculative_decode

def run_baseline(dataset, models, config):
    target_model, processor, device, dtype = models["target_model"], models["target_processor"], models["device"], models["dtype"]
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=config.language, task=config.task)
    total_time, predictions, references = 0.0, [], []
    
    for sample in dataset:
        audio = sample["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        inputs = {k: v.to(device, dtype=dtype if v.dtype.is_floating_point else v.dtype) for k, v in inputs.items()}
        
        with torch.no_grad():
            start = time.perf_counter()
            kwargs = {"forced_decoder_ids": forced_decoder_ids, "max_new_tokens": config.max_new_tokens}
            if config.top_p > 0: kwargs.update({"do_sample": True, "top_p": config.top_p, "temperature": config.temperature})
            else: kwargs.update({"do_sample": False})
            outputs = target_model.generate(**inputs, **kwargs)
            total_time += time.perf_counter() - start
        
        predictions.append(processor.batch_decode(outputs, skip_special_tokens=True)[0].strip().lower())
        references.append(sample["text"].strip().lower())
    
    return {"total_time": total_time, "avg_time": total_time / len(dataset), "predictions": predictions, "references": references}

def run_speculative(dataset, models, config):
    target_model, draft_model = models["target_model"], models["draft_model"]
    target_processor, draft_processor = models["target_processor"], models["draft_processor"]
    device, dtype = models["device"], models["dtype"]
    
    forced_decoder_ids = target_processor.get_decoder_prompt_ids(language=config.language, task=config.task)
    eos_token_id = target_model.config.eos_token_id
    total_time, predictions, references = 0.0, [], []
    
    for sample in dataset:
        audio = sample["audio"]
        target_inputs = target_processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        draft_inputs = draft_processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        
        target_inputs = {k: v.to(device, dtype=dtype if v.dtype.is_floating_point else v.dtype) for k, v in target_inputs.items()}
        draft_inputs = {k: v.to(device, dtype=dtype if v.dtype.is_floating_point else v.dtype) for k, v in draft_inputs.items()}
        
        with torch.no_grad():
            start = time.perf_counter()
            target_enc = target_model.get_encoder()(**target_inputs)
            draft_enc = draft_model.get_encoder()(**draft_inputs)
            
            prefix_ids = [target_model.config.decoder_start_token_id] + [fid[1] for fid in forced_decoder_ids]
            prefix = torch.tensor([prefix_ids], device=device, dtype=torch.long)
            
            output_tokens = speculative_decode(target_model, draft_model, target_enc, draft_enc, prefix, config.max_new_tokens, config.draft_k, eos_token_id, config.top_p, config.temperature)
            total_time += time.perf_counter() - start
        
        predictions.append(target_processor.batch_decode(output_tokens, skip_special_tokens=True)[0].strip().lower())
        references.append(sample["text"].strip().lower())
    
    return {"total_time": total_time, "avg_time": total_time / len(dataset), "predictions": predictions, "references": references}
