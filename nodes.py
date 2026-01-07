import comfy.utils

import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import uuid

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image, ImageOps

from .dreamidv_wan import DreamIDV
from .dreamidv_wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from .dreamidv_wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from .dreamidv_wan.utils.utils import cache_video, cache_image, str2bool

import cv2
import numpy as np
from .express_adaption.media_pipe import FaceMeshDetector, FaceMeshAlign_dreamidv
from .express_adaption.get_video_npy import get_video_npy
import folder_paths

def generate_pose_and_mask_videos(ref_video_path, ref_image_path):

    print("Starting online generation of pose and mask videos...")
    detector = FaceMeshDetector()
    get_align_motion = FaceMeshAlign_dreamidv()
    CORE_LANDMARK_INDICES = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327, 4, 24,
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
        468, 473, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107,
        300, 293, 334, 296, 336, 156,
    ]
    FACE_OVAL_INDICES = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]
    CORE_LANDMARK_INDICES.extend(FACE_OVAL_INDICES)
    CORE_LANDMARK_INDICES = list(set(CORE_LANDMARK_INDICES))
    def save_visualization_video(landmarks_sequence, output_filename, frame_size, fps=30, mode='points'):
        width, height = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_filename}")
            return
        print(f"Saving {mode} video to {output_filename}...")
        for frame_landmarks in landmarks_sequence:
            frame_image = np.zeros((height, width, 3), dtype=np.uint8)
            if mode == 'points':
                for landmark in frame_landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(frame_image, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
            elif mode == 'mask':
                face_oval_points = frame_landmarks.astype(np.int32)
                cv2.fillConvexPoly(frame_image, face_oval_points, color=(255, 255, 255))
            video_writer.write(frame_image)
        video_writer.release()
        print("Video saving complete.")
    fps = cv2.VideoCapture(ref_video_path).get(cv2.CAP_PROP_FPS)
    face_results = get_video_npy(ref_video_path)
    video_name = os.path.basename(ref_video_path).split('.')[0]
    #kiki:
    # temp_dir = os.path.join(os.path.dirname(ref_video_path), 'temp_generated')
    temp_dir = os.path.join(folder_paths.get_temp_directory(), 'dreamidv')
    os.makedirs(temp_dir, exist_ok=True)
    print(f'try open ref_image_path:{ref_image_path}')
    image = Image.open(ref_image_path).convert('RGB')
    ref_image = np.array(image)
    _, ref_img_lmk = detector(ref_image)
    _, pose_addvis = get_align_motion(face_results, ref_img_lmk)
    width, height = face_results[0]['width'], face_results[0]['height']
 
    pose_output_path = os.path.join(temp_dir, video_name + '_pose.mp4')
    core_landmarks_sequence = pose_addvis[:, CORE_LANDMARK_INDICES, :]
    save_visualization_video(
        landmarks_sequence=core_landmarks_sequence,
        output_filename=pose_output_path,
        frame_size=(width, height),
        fps=fps,
        mode='points'
    )
    mask_output_path = os.path.join(temp_dir, video_name + '_mask.mp4')
    face_oval_sequence = pose_addvis[:, FACE_OVAL_INDICES, :]
    save_visualization_video(
        landmarks_sequence=face_oval_sequence,
        output_filename=mask_output_path,
        frame_size=(width, height),
        fps=fps,
        mode='mask'
    )
    return pose_output_path, mask_output_path

class RunningHub_DreamID_V_Loader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"type": (["Wan2.2 I2V", "Wan2.1 T2V"], ),
            }
        }

    RETURN_TYPES = ('RH_DreamID-V_Pipeline', )
    RETURN_NAMES = ('DreamID-V Pipeline', )
    FUNCTION = "load"
    CATEGORY = "RunningHub/DreamID-V"

    OUTPUT_NODE = True

    def load(self, **kwargs):
        # hardcode
        task = 'swapface'
        ckpt_dir = os.path.join(folder_paths.models_dir, 'Wan', 'Wan2.1-T2V-1.3B')
        dreamidv_ckpt = os.path.join(folder_paths.models_dir, 'DreamID-V', 'dreamidv.pth')
        cfg = WAN_CONFIGS[task]
        wan_swapface = DreamIDV(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            dreamidv_ckpt=dreamidv_ckpt,
        )
        return (wan_swapface, )

class RunningHub_DreamID_V_Sampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"type": (["Wan2.2 I2V", "Wan2.1 T2V"], ),
                "pipeline": ("RH_DreamID-V_Pipeline", ),
                "video": ("VIDEO", ),
                "ref_image": ("IMAGE", ),
                # "width": ("INT", {"default": 832,}),
                # "height": ("INT", {"default": 480,}),
                "size": (["832*480", "1280*720"], {"default": "832*480"}),
                "frame_num": ("INT", {"default": 81, "min": 1, 'step': 4}),
                "sample_steps": ("INT", {"default": 20,}),
                "fps": ("INT", {"default": 24,}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('frames', )
    FUNCTION = "sample"
    CATEGORY = "RunningHub/DreamID-V"

    OUTPUT_NODE = True

    def tensor_2_pil(self,img_tensor):
        i = 255. * img_tensor.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def sample(self, **kwargs):

        #kiki hardcode
        sample_shift = 5.0
        sample_solver = 'unipc'
        sample_guide_scale_img = 4.0

        pipeline = kwargs.get('pipeline')
        print(pipeline.config)
        pipeline.config.sample_fps = kwargs.get('fps')
        print(pipeline.config)
        sample_steps = kwargs.get('sample_steps')
        self.pbar = comfy.utils.ProgressBar(sample_steps + 1)
        ref_video_path = kwargs.get('video').get_stream_source()
        ref_image = self.tensor_2_pil(kwargs.get('ref_image'))
        ref_image_path = os.path.join(folder_paths.get_temp_directory(), f'dreamidv_{uuid.uuid4()}.png')
        ref_image.save(ref_image_path)
        # width = kwargs.get('width')
        # height = kwargs.get('height')
        size = kwargs.get('size')
        seed = kwargs.get('seed') ^ (2 ** 32)
        frame_num = kwargs.get('frame_num')

        try:
            ref_pose_path, ref_mask_path = generate_pose_and_mask_videos(
                ref_video_path=ref_video_path,
                ref_image_path=ref_image_path
            )
        except:
            raise ValueError("Pose and mask video generation failed. no pose detected in the reference video.")
        text_prompt = 'change face'

        ref_paths = [
            ref_video_path,
            ref_mask_path,
            ref_image_path,
            ref_pose_path
        ]

        self.update()

        video = pipeline.generate(
            text_prompt,
            ref_paths,
            size=SIZE_CONFIGS[size],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale_img=sample_guide_scale_img,
            seed=seed,
            update_fn=self.update)
        print(f'video:{video.shape}')
        video = (video.clamp(-1, 1).cpu().permute(1, 2, 3, 0) + 1.0) / 2.0
        return (video, )

    def update(self):
        self.pbar.update(1)

NODE_CLASS_MAPPINGS = {
    "RunningHub_DreamID-V_Loader": RunningHub_DreamID_V_Loader,
    "RunningHub_DreamID-V_Sampler": RunningHub_DreamID_V_Sampler,
}