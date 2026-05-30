import torch
from transformers import AutoModel
import soundfile as sf
from pathlib import Path

def load_transformer_model(model_repo, model_dtype, device):
    model = AutoModel.from_pretrained(model_repo, trust_remote_code=True,
                                      attn_implementation="sdpa", dtype=model_dtype).to(device).to(model_dtype)
    model.tokenizer.quantizer.project_out.register_forward_pre_hook(cast_input_hook)
    return model

def cast_input_hook(module, args):
    return tuple(a.to(module.weight.dtype) if isinstance(a, torch.Tensor) else a for a in args)

def load_prompt_embedding_model_and_tokenizer(llm_repo_id, model_dtype, device):
    from transformers import AutoTokenizer
    prompt_embedding_model = AutoModel.from_pretrained(llm_repo_id, dtype=model_dtype).to(device).to(model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(llm_repo_id)
    return prompt_embedding_model, tokenizer

def load_audio(file_path, device, model_dtype):
    import torchaudio

    MAX_SAMPLES = 48000 * 60
    audio, sr = sf.read(file_path, always_2d=True)
    wav = torch.from_numpy(audio.T).float().to(device).to(model_dtype)

    if sr != 48000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)

    wav = wav[..., :MAX_SAMPLES]
    print(wav.shape)
    return wav

def get_reference_audio_latent(encoder, device, model_dtype, silence_latent, use_reference=True):
    ref_path = "inputs/reference1.wav"
    if use_reference and Path(ref_path).exists():
        wav = load_audio(ref_path, device, model_dtype)
        reference_latent = encoder.encode(wav).to(device).to(model_dtype)
        reference_latent = reference_latent.permute(0, 2, 1)
        mask = torch.LongTensor([0]).to(device)
        return reference_latent, mask
    print("No reference audio")
    return silence_latent.permute(0, 2, 1), torch.LongTensor([0]).to(device)

def load_finetuning_audio_latents(encoder, device, model_dtype, lim=-1):
    path = "inputs/lora"
    wavs = []

    for file in Path(path).iterdir():
        if file.is_file() and file.suffix == ".wav":
            audio_latent = load_audio(file, device, model_dtype)
            wavs.append(audio_latent)
    if lim != -1:
        wavs = wavs[:lim]
    wavs = torch.stack(wavs).to(device).to(model_dtype)
    print(wavs.shape)
    latents = encoder.encode(wavs).to(device).to(model_dtype)
    return latents

def load_encoder(config_path, checkpoint_path, device, model_dtype):
    from stable_audio_vae import StableAudioVAE

    return StableAudioVAE(config_path=config_path,checkpoint_path=checkpoint_path).to(device).to(model_dtype)

def load_silence_latent(model_repo, filename, device, model_dtype):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model_repo, filename=filename)
    silence_latent = torch.load(path).to(device).to(model_dtype)
    return silence_latent

def save_audio(audio_data, filename, sample_rate=48000):

    output_path = Path("./outputs/" + filename)
    audio_np = audio_data.to(torch.float32).cpu().detach().numpy().T.reshape((-1, 2))
    sf.write(output_path, audio_np, sample_rate)
    return audio_np

def decode_latent_and_save_audio(latents, decoder, filename, sample_rate=48000):
    output = decoder.decode(latents)
    print(output.shape)
    audio_np = save_audio(output, filename, sample_rate)
    print(audio_np.shape)