from fast_whisper.transcriber import FastWhisperTranscriber as SpeculativeWhisper

# Initialize the model 
# Note: We use 'openai/whisper-tiny' and 'openai/whisper-large-v3' as defaults in the class,
# but we can specify them explicitly here.
sw = SpeculativeWhisper(draft_model="tiny", target_model="large-v3", device="cpu") 
# Using CPU for local demo, change to "cuda" if you have a GPU

audio_files = ["audio1.wav", "audio2.wav"]

print("Transcribing...")
# The transcribe method accepts a list and returns a list of texts
outputs = sw.transcribe(audio_files, max_tokens=200, batch_size=2)

for audio, text in zip(audio_files, outputs):
    print(f"{audio}: {text}")
