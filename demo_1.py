import logging
logging.basicConfig(level=logging.ERROR)

from pathlib import Path
import torch

from mmaudio.eval_utils import ModelConfig, all_model_cfg, generate, load_video
from tqdm import tqdm
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils
import librosa
import time
import os

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading models...')
with torch.inference_mode():
    model: ModelConfig = all_model_cfg['large_44k_v2']
    model.download_if_needed()
    seq_cfg = model.seq_cfg
    device = 'cuda'
    dtype = torch.bfloat16

    # load a pretrained model
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=25)
    
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=True
    )
    feature_utils = feature_utils.to(device, dtype).eval()
    text_encoded = feature_utils.encode_text([''])

@torch.inference_mode()
def main(video: str):
    video_path: Path = Path(video).expanduser()
    video_info = load_video(video_path, 5.0)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)

    if seq_cfg.duration != video_info.duration_sec:
        seq_cfg.duration = video_info.duration_sec
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(
        clip_frames,
        sync_frames, 
        text_encoded,
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=4.5
    )
    audio = audios.float().cpu()[0].squeeze().numpy()
    stretched = librosa.effects.time_stretch(audio, rate=fps/25)
    stretched_tensor = torch.from_numpy(stretched).unsqueeze(1)

if __name__ == '__main__':
    video = '/home/eugene/Downloads/test_mmaudio/1.mp4'
    avg = 0
    
    iterations = 50
    name = video.split("/")[-1]
    for _ in tqdm(range(iterations), total=iterations, desc=name, colour='green'):
        t1 = time.time()
        main(video)
        t2 = time.time()
        avg += t2-t1
    
    print(f'{name} - Time: {round(avg/iterations, 4)}s')

# 2.0505s