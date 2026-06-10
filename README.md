# Audio Diffusion Finetuning

LoRA finetuning for [ACE-Step](https://huggingface.co/ACE-Step/acestep-v15-turbo-shift1), a diffusion-based audio generation model. Adds low-rank adapters to the transformer decoder's attention layers and trains on your own audio data.

## How it works

- Encodes training audio into VAE latent space using Stable Audio VAE
- Freezes the base ACE-Step transformer and attaches LoRA matrices (rank 8) to Q/K/V/O projections in the second half of decoder layers
- Trains with flow-matching velocity prediction (MSE loss, AdamW optimizer)
- After training, merges LoRA weights back into the base model and generates audio

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA 12.6 (tested on Windows)
- [uv](https://docs.astral.sh/uv/) package manager
- ~16GB+ VRAM recommended (encoding is batched to fit in limited VRAM)

## Setup

1. Clone the repo
2. Install dependencies:
   ```
   uv sync
   ```

## Usage

1. Place your `.wav` training files in `./inputs/lora/`
2. Open `lora.ipynb` and run cells top to bottom
3. Output audio is saved to `./outputs/`

### Training parameters

Defaults in the notebook:

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| Learning rate | 1e-3 |
| Epochs | 30 |
| Batch size | 1 |
| Audio segment length | 60 seconds |
| Sample rate | 48kHz (stereo) |

Checkpoints are saved to `./checkpoints/checkpoint.pt`.

## Project structure

```
lora.ipynb           # main training notebook
model_utils.py       # model loading, audio I/O, VAE encoding/decoding
stable_audio_vae.py  # VAE wrapper for Stable Audio Tools
models/              # VAE config + checkpoint
inputs/lora/         # training audio (not tracked)
outputs/             # generated audio (not tracked)
checkpoints/         # finetuned model weights (not tracked)
```
