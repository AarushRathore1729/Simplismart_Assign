"""High-level transcription API for fast Whisper decoding."""

import torch
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
from .settings import Config
from .model_loader import load_models
from .decoder import speculative_decode

class FastWhisperTranscriber:
    MODEL_MAP = {
        "tiny": "openai/whisper-tiny", "base": "openai/whisper-base",
        "small": "openai/whisper-small", "medium": "openai/whisper-medium",
        "large": "openai/whisper-large", "large-v2": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3",
    }
    
    def __init__(self, draft_model: str = "tiny", target_model: str = "large-v3", device: Optional[str] = None,
                 draft_k: int = 6, top_p: float = 0.0, temperature: float = 1.0, language: str = "en", task: str = "transcribe"):
        self.draft_k, self.top_p, self.temperature = draft_k, top_p, temperature
        self.language, self.task = language, task
        
        config = Config(
            target_model=self.MODEL_MAP.get(target_model, target_model),
            draft_model=self.MODEL_MAP.get(draft_model, draft_model),
            device=device, language=language, task=task,
        )
        
        print(f"Loading models: {draft_model} (draft) + {target_model} (target)...")
        self.models = load_models(config)
        self.device, self.dtype = self.models["device"], self.models["dtype"]
        print(f"Ready on {self.device}")
    
    def transcribe(self, audio: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
                   max_tokens: int = 128, batch_size: int = 1, return_timing: bool = False) -> Union[str, List[str], dict]:
        import time, soundfile as sf
        
        audio_list = [audio] if not isinstance(audio, list) else audio
        target_model, draft_model = self.models["target_model"], self.models["draft_model"]
        target_proc, draft_proc = self.models["target_processor"], self.models["draft_processor"]
        
        forced_decoder_ids = target_proc.get_decoder_prompt_ids(language=self.language, task=self.task)
        eos_token_id = target_model.config.eos_token_id
        
        results, total_time = [], 0.0
        
        for audio_input in audio_list:
            if isinstance(audio_input, (str, Path)):
                audio_array, sample_rate = sf.read(str(audio_input))
                if sample_rate != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            else:
                audio_array = audio_input
            
            t_in = {k: v.to(self.device, dtype=self.dtype if v.dtype.is_floating_point else v.dtype) 
                    for k, v in target_proc(audio_array, sampling_rate=16000, return_tensors="pt").items()}
            d_in = {k: v.to(self.device, dtype=self.dtype if v.dtype.is_floating_point else v.dtype) 
                    for k, v in draft_proc(audio_array, sampling_rate=16000, return_tensors="pt").items()}
            
            with torch.no_grad():
                start = time.perf_counter()
                target_enc = target_model.get_encoder()(**t_in)
                draft_enc = draft_model.get_encoder()(**d_in)
                
                prefix = torch.tensor([[target_model.config.decoder_start_token_id] + [fid[1] for fid in forced_decoder_ids]], 
                                      device=self.device, dtype=torch.long)
                
                output_tokens = speculative_decode(
                    target_model, draft_model, target_enc, draft_enc, prefix, max_tokens, 
                    self.draft_k, eos_token_id, self.top_p, self.temperature
                )
                total_time += time.perf_counter() - start
            
            results.append(target_proc.batch_decode(output_tokens, skip_special_tokens=True)[0].strip())
        
        if return_timing:
            return {"texts": results if isinstance(audio, list) else results[0], 
                    "total_time": total_time, "avg_time": total_time / len(audio_list)}
        return results if isinstance(audio, list) else results[0]
    
    def __repr__(self):
        return f"FastWhisperTranscriber(draft_k={self.draft_k}, top_p={self.top_p}, device={self.device})"
