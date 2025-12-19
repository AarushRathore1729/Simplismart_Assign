import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

def load_models(config):
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    target_proc = AutoProcessor.from_pretrained(config.target_model)
    draft_proc = AutoProcessor.from_pretrained(config.draft_model)
    
    def _load(name):
        return AutoModelForSpeechSeq2Seq.from_pretrained(
            name, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device).eval()

    target_model, draft_model = _load(config.target_model), _load(config.draft_model)
    target_model.config.use_cache = draft_model.config.use_cache = True
    
    return {
        "target_model": target_model, "draft_model": draft_model,
        "target_processor": target_proc, "draft_processor": draft_proc,
        "device": device, "dtype": dtype
    }
