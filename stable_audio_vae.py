import os
import json
import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm, weight_norm
import torchaudio

from stable_audio_tools.models.autoencoders import create_autoencoder_from_config


DEFAULT_ROOT = "./"
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_ROOT, "config.json")
DEFAULT_CHECKPOINT_PATH = os.path.join(DEFAULT_ROOT, "checkpoint.ckpt")


def remove_weight_norm_(module):
    """Recursively remove weight normalization from all submodules."""
    for name, child in module.named_children():
        if hasattr(child, "weight"):
            try:
                remove_weight_norm(child)
            except ValueError:
                pass
        remove_weight_norm_(child)


def add_weight_norm_(module):
    """Recursively add weight normalization to all submodules."""
    for name, child in module.named_children():
        if hasattr(child, "weight"):
            weight_norm(child)
        add_weight_norm_(child)


def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    """Resample, pad/crop, and set audio channels."""
    audio = audio.to(device)

    if in_sr != target_sr:
        audio = torchaudio.functional.resample(
            audio, orig_freq=in_sr, new_freq=target_sr
        )
    if target_length is None:
        target_length = audio.shape[-1]
    audio = PadCrop(target_length, randomize=False)(audio)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)
    return audio


class PadCrop(torch.nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(
            0, max(0, s - self.n_samples) + 1, []
        ).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output


def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


class StableAudioVAE(nn.Module):
    def __init__(
        self,
        sampling_rate=48000,
        config_path=DEFAULT_CONFIG_PATH,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        scale_factor=1.0,
        shift_factor=0.0,
        remove_norm=False,
        overlap=32,
        chunk_size=128,
    ):
        super(StableAudioVAE, self).__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.vae = create_autoencoder_from_config(self.config)

        # Load checkpoint - support both .ckpt (PyTorch) and .safetensors
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            checkpoints = load_file(checkpoint_path)
        else:
            checkpoints = torch.load(
                checkpoint_path, map_location=torch.device("cpu")
            )
            if "state_dict" in checkpoints:
                checkpoints = checkpoints["state_dict"]

        # Strip "autoencoder." prefix if present
        has_autoencoder = any(
            k.startswith("autoencoder.") for k in checkpoints.keys()
        )
        if has_autoencoder:
            checkpoints = {
                k.replace("autoencoder.", ""): v
                for k, v in checkpoints.items()
                if k.startswith("autoencoder.")
            }
        self.vae.load_state_dict(checkpoints)

        if remove_norm:
            remove_weight_norm_(self.vae)

        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        self.sampling_rate = sampling_rate
        self.io_channels = self.config["audio_channels"]
        self.overlap = overlap
        self.chunk_size = chunk_size
        self.downsampling_ratio = self.vae.downsampling_ratio
        self.latent_dim = self.vae.latent_dim

    def load_wav(self, path):
        wav, sr = torchaudio.load(path)
        wav = prepare_audio(
            wav,
            in_sr=sr,
            target_sr=self.sampling_rate,
            target_length=None,
            target_channels=self.io_channels,
            device="cpu",
        )
        return wav

    @torch.no_grad()
    def encode(self, wav, chunked=False):
        if wav.shape[1] <= self.chunk_size * self.vae.downsampling_ratio:
            chunked = False
        latent = self.vae.encode_audio(wav, chunked=chunked)
        latent = self.scale_factor * (latent - self.shift_factor)
        return latent

    @torch.no_grad()
    def decode(self, z, chunked=False):
        z = z / self.scale_factor + self.shift_factor
        if z.shape[-1] <= self.chunk_size:
            chunked = False
        output = self.vae.decode_audio(z, chunked=chunked)
        return output

    @torch.no_grad()
    def forward(self, wav, chunked=False):
        """Encode and decode audio (reconstruction)."""
        latent = self.vae.encode_audio(wav, chunked=chunked)
        latent = self.scale_factor * (latent - self.shift_factor)
        latent = latent / self.scale_factor + self.shift_factor
        output = self.vae.decode_audio(latent, chunked=chunked)
        return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode and decode audio with StableAudioVAE")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_CHECKPOINT_PATH, help="path to checkpoint")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.json")
    parser.add_argument("-i", "--input", type=str, required=True, help="input audio file")
    parser.add_argument("-o", "--output", type=str, required=True, help="output audio file")
    parser.add_argument("-sr", "--sampling_rate", type=int, default=48000, help="sampling rate")
    parser.add_argument("--chunked", action="store_true", help="use chunked processing for long audio")
    args = parser.parse_args()

    pipeline = StableAudioVAE(
        sampling_rate=args.sampling_rate,
        config_path=args.config,
        checkpoint_path=args.model,
    )
    pipeline = pipeline.cuda()

    wav = pipeline.load_wav(args.input)
    wav = wav.cuda()
    print(f"Input shape: {wav.shape}")

    z = pipeline.encode(wav, chunked=args.chunked)
    print(f"Latent shape: {z.shape}")

    output = pipeline.decode(z, chunked=args.chunked)
    print(f"Output shape: {output.shape}")

    output = output[0].cpu()
    torchaudio.save(args.output, output, pipeline.sampling_rate)
    print(f"Saved to {args.output}")
