import os
import argparse
import itertools
import time
import numpy as np
import io
from pathlib import Path
import random
from safetensors import safe_open

import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image

from ip_adapter import StyleShot
from ip_adapter.style_encoder import Style_Aware_Encoder
from ip_adapter.utils import is_torch2_available
from ip_adapter.tools import pre_processing
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


base_model_path = "runwayml/stable-diffusion-v1-5"
image_encoder_path = "/disk1/zl/InstantStyle/models/image_encoder"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"

# load model
transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
styleshot_model_path = "Gaojunyao/StyleShot"
device = "cuda"
if not os.path.isdir(base_model_path):
    base_model_path = snapshot_download(base_model_path, local_dir=base_model_path)
    print(f"Downloaded model to {base_model_path}")
if not os.path.isdir(transformer_block_path):
    transformer_block_path = snapshot_download(transformer_block_path, local_dir=transformer_block_path)
    print(f"Downloaded model to {transformer_block_path}")
if not os.path.isdir(styleshot_model_path):
    styleshot_model_path = snapshot_download(styleshot_model_path, local_dir=styleshot_model_path)
    print(f"Downloaded model to {styleshot_model_path}")

ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
style_aware_encoder_path = os.path.join(styleshot_model_path, "pretrained_weight/style_aware_encoder.bin")
pipe = StableDiffusionPipeline.from_pretrained(base_model_path)

# load StyleShot
ip_model = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path, less_condition=True)

state_dict = {}
with safe_open("/disk1/zl/StyleShot-main/sd-ip_adapter_ship/checkpoint-2000/model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)
ip_model.feature_adapter_model.load_state_dict(state_dict)
ip_model.feature_adapter_model.to(dtype=pipe.dtype)

from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer

image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to("cuda", dtype=torch.float16)

clip_image_processor = CLIPImageProcessor()

style_aware_encoder = Style_Aware_Encoder(image_encoder)
# if args.pretrained_style_encoder_path is not None:
#     style_aware_encoder.load_state_dict(torch.load(args.pretrained_style_encoder_path))

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        default="/disk1/zl/InstantStyle/data/object",
        help="Training data root path",
    )
    parser.add_argument(
        "--less_condition",
        default=False,
        action='store_true',
        help="Use image prompt adapter when less_condition is True, otherwise use text prompt adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--content_prompt",
        type=str,
        default="ship",
        help="The content in style reference.",
    )
    parser.add_argument(
        "--pretrained_style_encoder_path",
        type=str,
        default=None,
        help="Path to pretrained style-aware encoder. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=2000)
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    args = parser.parse_args()

    return args


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_root_path="data/object", content_prompt="ship"):
        super().__init__()
        self.none_loop = 0
        # self.tokenizer = tokenizer  # CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        self.image_root_path = image_root_path
        self.content_prompt = content_prompt
        self.content_prompt_ = content_prompt

        self.data = os.listdir(os.path.join(self.image_root_path, self.content_prompt_, self.content_prompt_))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        ])

    def load_data(self, item):
        try:
            text = item["content_prompt"]  # the content prompt to be removed
        except Exception as e:
            print(e)
            text = ""
        image_file = item["image_file"]  # file name
        try:
            raw_image = Image.open(os.path.join(self.image_root_path, self.content_prompt_, self.content_prompt_, image_file))
        except Exception as e:
            print(e)
            raw_image = None
        return raw_image, text, image_file

    def __getitem__(self, idx):
        item = {"image_file": self.data[idx], "content_prompt": self.content_prompt}
        raw_image, text, image_file = self.load_data(item)
        image = raw_image.resize((512, 512))  # self.transform(raw_image.convert("RGB"))

        return {
            "image_file": image_file,
            "image": image,
            "text": text,
            "raw_image": raw_image,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = [example["image"] for example in data]  # torch.stack([example["image"] for example in data])
    texts = [example["text"] for example in data]
    image_files = [example["image_file"] for example in data]
    raw_images = [example["raw_image"].convert("RGB").resize((512, 512)) for example in data]
    return {
        "image_files": image_files,
        "images": images,
        "texts": texts,
        "raw_images": raw_images,
    }


def model(image, classname, idx, args):
    print(args.less_condition)
    images = ip_model.generate(style_image=image,
                    prompt=[["A person"]],  # "best quality, high quality",    #
                    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, "
                                    "low contrast, noisy, saturation, blurry",
                    scale=1.0,
                    guidance_scale=7.5,
                    num_samples=1,
                    num_inference_steps=30,
                    seed=42,
                    neg_content_name=classname,
                    neg_content_scale=0.6,
                    less_condition=args.less_condition,
                  )

    if args.less_condition:
        os.makedirs("outputs/ship/ours_person-2000", exist_ok=True)
        images[0][0].save("outputs/ship/ours_person-2000/ours_{}.png".format(idx))
    else:
        os.makedirs("outputs/ship/base_person-2000", exist_ok=True)
        images[0][0].save("outputs/ship/base_person-2000/base_{}.png".format(idx))


def main():
    args = parse_args()
    # dataloader
    train_dataset = MyDataset(image_root_path=args.image_root_path, content_prompt=args.content_prompt)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    image_encoder.requires_grad_(False)
    # ip_model.image_proj_model.requires_grad_(False)
    # ip_model.image_encoder.requires_grad_(False)
    ip_model.pipe.unet.requires_grad_(False)

    for i, batch in enumerate(train_dataloader):
        model(batch["images"][0], batch["texts"][0], batch["image_files"][0].split(".")[0], args)


if __name__ == "__main__":
    main()











