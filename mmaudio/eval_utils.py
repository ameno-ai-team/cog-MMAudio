import dataclasses
from pathlib import Path
from typing import Optional, Any

import torch
from torchvision.transforms import v2

from mmaudio.data.av_utils import VideoInfo, read_frames
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio
from mmaudio.model.sequence_config import CONFIG_44K, SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.utils.download_utils import download_model_if_needed
import time

@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_path: Path
    vae_path: Path
    bigvgan_16k_path: Optional[Path]
    mode: str
    synchformer_ckpt: Path = Path('./ext_weights/synchformer_state_dict.pth')

    @property
    def seq_cfg(self) -> SequenceConfig:
        return CONFIG_44K

    def download_if_needed(self):
        download_model_if_needed(self.model_path)
        download_model_if_needed(self.vae_path)
        download_model_if_needed(self.synchformer_ckpt)

large_44k_v2 = ModelConfig(model_name='large_44k_v2',
                           model_path=Path('./weights/mmaudio_large_44k_v2.pth'),
                           vae_path=Path('./ext_weights/v1-44.pth'),
                           bigvgan_16k_path=None,
                           mode='44k')
all_model_cfg: dict[str, ModelConfig] = { 'large_44k_v2': large_44k_v2 }

def generate(
    clip_video: torch.Tensor,
    sync_video: torch.Tensor,
    text_encoded: Any,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
) -> torch.Tensor:
    device = feature_utils.device
    dtype = feature_utils.dtype

    clip_video = clip_video.to(device, dtype, non_blocking=True)
    clip_features = feature_utils.encode_video_with_clip(clip_video, batch_size=40)

    sync_video = sync_video.to(device, dtype, non_blocking=True)
    sync_features = feature_utils.encode_video_with_sync(sync_video, batch_size=40)

    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_encoded)
    empty_conditions = net.get_empty_conditions(1, negative_text_features=text_encoded)

    x0 = torch.randn(1, net.latent_seq_len, net.latent_dim, device=device, dtype=dtype, generator=rng)
    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions, cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio

_CLIP_TRANSFORM = v2.Compose([
    v2.Resize((384, 384), interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

_SYNC_TRANSFORM = v2.Compose([
    v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def load_video_comfy(images, duration_sec: float) -> VideoInfo:
    num_frames = len(images)
    
    output_indices = [[], []]
    target_fps = [8, 25]
    
    for frame_idx in range(num_frames):
        frame_time = frame_idx * 0.04
        
        for i in range(2):
            expected_frame_idx = int(frame_time * target_fps[i])
            if expected_frame_idx >= len(output_indices[i]):
                output_indices[i].append(frame_idx)
    
    all_frames = images.cpu().numpy()
    output_frames = [all_frames[indices] for indices in output_indices]

    clip_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2).to('cuda')
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2).to('cuda')

    clip_frames = _CLIP_TRANSFORM(clip_chunk)
    sync_frames = _SYNC_TRANSFORM(sync_chunk)

    clip_length_sec = clip_frames.shape[0] / 8
    sync_length_sec = sync_frames.shape[0] / 25

    if clip_length_sec < duration_sec:
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(8 * duration_sec)]
    sync_frames = sync_frames[:int(25 * duration_sec)]

    video_info = VideoInfo(
        duration_sec=duration_sec,
        fps=25,
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        all_frames=all_frames
    )
    return video_info

def load_video(video_path: Path, duration_sec: float) -> VideoInfo:
    output_frames, all_frames = read_frames(video_path, duration_sec)

    clip_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2).to('cuda')
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2).to('cuda')

    clip_frames = _CLIP_TRANSFORM(clip_chunk)
    sync_frames = _SYNC_TRANSFORM(sync_chunk)

    clip_length_sec = clip_frames.shape[0] / 8.0
    sync_length_sec = sync_frames.shape[0] / 25.0

    if clip_length_sec < duration_sec:
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(8.0 * duration_sec)]
    sync_frames = sync_frames[:int(25.0 * duration_sec)]

    video_info = VideoInfo(
        duration_sec=duration_sec,
        fps=25,
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        all_frames=all_frames
    )
    return video_info