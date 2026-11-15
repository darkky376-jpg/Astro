---
name: omnivoice-tts
description: Generate speech audio with OmniVoice — zero-shot voice cloning from a short reference clip, voice design from speaker attributes, or auto voice. Supports 600+ languages, non-verbal tags, and pronunciation control. Use when the user wants to convert text to speech, clone a voice, design a custom voice, or generate audio/voiceover for videos.
version: 1.0.0
---

# OmniVoice TTS Skill

OmniVoice ([k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)) is a massively multilingual zero-shot TTS model supporting 600+ languages, with voice cloning, voice design, and fast inference (RTF as low as 0.025).

## Setup

Requires Python 3.10+ and PyTorch installed first.

```bash
# Recommended: use a fresh virtual environment (conda/venv)

# Step 1: PyTorch (NVIDIA GPU example; see https://pytorch.org/get-started/locally/ for your platform)
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# Apple Silicon: pip install torch==2.8.0 torchaudio==2.8.0
# Intel Arc GPU (XPU): pip install torch torchaudio --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Step 2: OmniVoice
pip install omnivoice          # stable release from PyPI
# or: pip install git+https://github.com/k2-fsa/OmniVoice.git

# Optional extras
pip install "omnivoice[tn]"    # text normalization (numbers read correctly); macOS needs: conda install -c conda-forge pynini
pip install flashinfer-python==0.6.15.post1 "flashinfer-jit-cache==0.6.15.post1+cu128" --extra-index-url https://flashinfer.ai/whl/cu128/  # ~2-2.9x faster inference (NVIDIA)
```

If HuggingFace downloads are blocked, set: `export HF_ENDPOINT="https://hf-mirror.com"`

## Usage Modes

OmniVoice supports three generation modes via a shared `model.generate()` API.

### 1. Voice Cloning (most stable)

Clone a voice from a 3–10 second reference audio clip:

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
# Apple Silicon: device_map="mps" | Intel Arc GPU: device_map="xpu" | CPU: device_map="cpu"

audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    ref_text="Transcription of the reference audio.",  # can be omitted -> Whisper auto-transcribes
)
sf.write("out.wav", audio[0], 24000)  # output is 24 kHz
```

Reuse a cloned voice across sessions (encode the reference once):

```python
prompt = model.create_voice_clone_prompt(ref_audio="ref.wav", ref_text="Transcription of the reference audio.")
prompt.save("my_voice.pt")

# Later:
from omnivoice import VoiceClonePrompt
prompt = VoiceClonePrompt.load("my_voice.pt")
audio = model.generate(text="Hello again!", voice_clone_prompt=prompt)
```

Cloning tips:
- Use a 3–10 second reference clip; longer audio slows inference and degrades quality.
- Same-language reference audio gives standard pronunciation; cross-lingual cloning carries the reference language's accent.
- Use `normalize_text=True` (with `omnivoice[tn]` installed) so "123" reads as "one hundred twenty-three".

### 2. Voice Design (no reference audio needed)

Describe the desired voice with comma-separated attributes: **gender** (male/female), **age** (child to elderly), **pitch** (very low to very high), **style** (whisper), **English accent** (American, British, etc.), **Chinese dialect** (四川话, 陕西话, etc.):

```python
audio = model.generate(
    text="Hello, this is a test of zero-shot voice design.",
    instruct="female, low pitch, british accent",
)
```

Note: voice design is trained on Chinese and English data; other languages may be less stable.

### 3. Auto Voice (random voice)

```python
audio = model.generate(text="This is a sentence without any voice prompt.")
```

## Generation Parameters

```python
audio = model.generate(
    text="...",
    num_step=32,   # diffusion steps (16 for faster inference)
    speed=1.0,     # >1.0 faster, <1.0 slower
    duration=10.0,  # fixed output duration in seconds (overrides speed)
)
```

## Expressive Control (inline in text)

Non-verbal tags: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`

```python
audio = model.generate(text="[laughter] You really got me. I didn't see that coming at all.")
```

Pronunciation correction:
- Chinese: pinyin with tone numbers in the text — `model.generate(text="这批货物打ZHE2出售后他严重SHE2本了。")`
- English: CMU dict uppercase phonemes in brackets — `model.generate(text="He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.")`

## Command-Line Tools

```bash
# Interactive Gradio web demo
omnivoice-demo --ip 0.0.0.0 --port 8001

# Single inference — voice cloning
omnivoice-infer --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --ref_audio ref.wav \
    --ref_text "Transcription of the reference audio." \
    --output hello.wav

# Single inference — voice design
omnivoice-infer --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --instruct "male, British accent" \
    --output hello.wav

# Batch inference across multiple GPUs (test.jsonl: one JSON object per line; only "id" and "text" mandatory)
omnivoice-infer-batch --model k2-fsa/OmniVoice --test_list test.jsonl --res_dir results/
```

JSONL fields: `id`, `text`, `ref_audio`+`ref_text` (cloning), `instruct` (design), optional `language_id`, `duration`, `speed`.

## FlashInfer Acceleration (optional)

```python
from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
apply_flashinfer(model)                          # throughput / batched inference
apply_flashinfer(model, enable_cuda_graph=True)  # recommended for batch=1 (low latency)
```

## Reference

- Repo: https://github.com/k2-fsa/OmniVoice
- Model: https://huggingface.co/k2-fsa/OmniVoice
- Web demo: https://huggingface.co/spaces/k2-fsa/OmniVoice
- Tips: docs/tips.md · Voice design: docs/voice-design.md · Parameters: docs/generation-parameters.md (in repo)
- Issues: https://github.com/k2-fsa/OmniVoice/issues
