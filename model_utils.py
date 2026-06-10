import torch
from transformers import AutoModel
import soundfile as sf
from pathlib import Path


# cast inputs to match module weight dtype (fixes bf16/fp32 mismatch in quantizer)
def cast_input_hook(module, args):
    return tuple(a.to(module.weight.dtype) if isinstance(a, torch.Tensor) else a for a in args)


# load the main DiT and fix dtype mismatch in quantizer output projection
def load_transformer_model(model_repo, model_dtype, device):
    model = AutoModel.from_pretrained(model_repo, trust_remote_code=True,
                                      attn_implementation="sdpa", dtype=model_dtype).to(device).to(model_dtype)
    model.tokenizer.quantizer.project_out.register_forward_pre_hook(cast_input_hook)
    return model


# load LLM for encoding text prompts into embeddings
def load_prompt_embedding_model_and_tokenizer(llm_repo_id, model_dtype, device):
    from transformers import AutoTokenizer
    prompt_embedding_model = AutoModel.from_pretrained(llm_repo_id, dtype=model_dtype).to(device).to(model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(llm_repo_id)
    return prompt_embedding_model, tokenizer


# load audio file, convert to tensor, resample to 48kHz if needed
def load_audio(file_path, device, model_dtype):
    import torchaudio

    audio, sr = sf.read(file_path, always_2d=True)
    wav = torch.from_numpy(audio.T).float().to(device).to(model_dtype)

    if sr != 48000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)

    return wav


# encode reference audio to latent space, or fall back to silence
def get_reference_audio_latent(ref_path, encoder, device, model_dtype, silence_latent, use_reference=True):
    if use_reference and Path(ref_path).exists():
        MAX_SAMPLES = 48000 * 60
        wav = load_audio(ref_path, device, model_dtype)
        wav = wav[..., :MAX_SAMPLES]
        reference_latent = encoder.encode(wav).to(device).to(model_dtype)
        reference_latent = reference_latent.permute(0, 2, 1)
        mask = torch.LongTensor([0]).to(device)
        return reference_latent, mask
    print("No reference audio")
    return silence_latent.permute(0, 2, 1), torch.LongTensor([0]).to(device)


# list all .wav files in a directory, sorted
def get_files_in_path_as_array(path):
    files = sorted([f for f in Path(path).iterdir() if f.is_file() and f.suffix == ".wav"])
    return files


# slice audio files into 60s segments and stack into one tensor
def load_finetuning_audio_segments(files, device, model_dtype):
    SEGMENT_SAMPLES = 48000 * 60
    wavs = []

    for file in files:
        audio = load_audio(file, device, model_dtype)
        num_segments = audio.shape[-1] // SEGMENT_SAMPLES
        for i in range(num_segments):
            segment = audio[..., i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES]
            wavs.append(segment)

    return torch.stack(wavs).to(device).to(model_dtype)


# run VAE encoder on a batch of waveforms
def encode_audio_segments(encoder, wavs, device, model_dtype):
    return encoder.encode(wavs).to(device).to(model_dtype)


# load the Stable Audio VAE encoder/decoder
def load_encoder(config_path, checkpoint_path, device, model_dtype):
    from stable_audio_vae import StableAudioVAE
    return StableAudioVAE(config_path=config_path, checkpoint_path=checkpoint_path).to(device).to(model_dtype)


# download pre-computed silence latent from HuggingFace Hub
def load_silence_latent(model_repo, filename, device, model_dtype):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model_repo, filename=filename)
    silence_latent = torch.load(path).to(device).to(model_dtype)
    return silence_latent


# convert tensor to numpy and write wav file
def save_audio(audio_data, filename, sample_rate=48000, output_dir="./outputs/"):
    output_path = Path(output_dir) / filename
    audio_np = audio_data.to(torch.float32).cpu().detach().numpy().T.reshape((-1, 2))
    sf.write(output_path, audio_np, sample_rate)
    print("Audio saved to " + str(output_path))
    return audio_np


# decode a batch of latents and save each as a wav
def decode_latent_and_save_audio(latents, decoder, filename, sample_rate=48000):
    output = decoder.decode(latents)
    print("decoded shape:", output.shape)
    for i in range(output.shape[0]):
        audio_np = save_audio(output[i:i+1,:,:], filename + str(i) + ".wav", sample_rate)
        print("saved:", audio_np.shape)