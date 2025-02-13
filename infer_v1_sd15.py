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

from ip_adapter import IPAdapterXL, AutoIPAdapterXL, AutoIPAdapterZS
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


base_model_path = "runwayml/stable-diffusion-v1-5"
image_encoder_path = "models/image_encoder"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] # for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
# target_blocks = [ "down_blocks.2.attentions.1"] # for layout blocks
# ip_model = AutoIPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks)
# ip_model = AutoIPAdapter(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])
# ip_model = AutoIPAdapter(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"])

# ip_model = AutoIPAdapter(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["block"], less_condition=True)


from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer

image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to("cuda", dtype=torch.float16)

clip_image_processor = CLIPImageProcessor()


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
        default="data/object",
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
        "--method",
        type=str,
        default="ours",
    )
    parser.add_argument(
        "--target_content",
        type=str,
        default="deer",
        help="The target content.",
    )
    parser.add_argument(
        "--load_epoch",
        type=int,
        default=500,
        help="The epoch of the trained model.",
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
    parser.add_argument("--num_train_epochs", type=int, default=100)
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
    # guidance_scale
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--neg_content_scale",
        type=float,
        default=0.8,
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
        self.content_prompt = content_prompt  #
        self.content_prompt_ = content_prompt

        self.data = os.listdir(os.path.join(self.image_root_path, self.content_prompt_, self.content_prompt_))
        # self.data = [data for data in self.data if "snowy" in data]

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


def model(image, neg_content_embd, idx, args, ip_model):
    print(args.less_condition)
    if args.target_content in ["airplane", "automobile"]:
        target_content = "An " + args.target_content
    else:
        target_content = "A " + args.target_content

    neg_content_scale = 1.0 if args.less_condition else args.neg_content_scale
    neg_content_scale = 0.0 if args.method == "base0" else neg_content_scale

    images = ip_model.generate(pil_image=image,
                    prompt="{}".format(target_content),
                    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, "
                                    "low contrast, noisy, saturation, blurry",
                    scale=args.scale,
                    guidance_scale=7.5,
                    num_samples=4,
                    num_inference_steps=30,
                    seed=0,
                    neg_content_embd=neg_content_embd,
                    neg_content_scale=neg_content_scale,
                    less_condition=args.less_condition,
                    theta=0.7,
                  )

    if "test" not in args.image_root_path:
        os.makedirs("outputs_zs{}/clip_encoder_zs_{}/{}/{}/{}".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method), exist_ok=True)
        images[0].save("outputs_zs{}/clip_encoder_zs_{}/{}/{}/{}/{}_0.png".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method, idx))
        images[1].save("outputs_zs{}/clip_encoder_zs_{}/{}/{}/{}/{}_1.png".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method, idx))
        images[2].save("outputs_zs{}/clip_encoder_zs_{}/{}/{}/{}/{}_2.png".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method, idx))
        images[3].save("outputs_zs{}/clip_encoder_zs_{}/{}/{}/{}/{}_3.png".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method, idx))
    else:
        os.makedirs("outputs_zs{}/clip_encoder_zs_{}_test/{}/{}/{}".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method), exist_ok=True)
        images[0].save("outputs_zs{}/clip_encoder_zs_{}_test/{}/{}/{}/{}_0.png".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method, idx))
        images[1].save("outputs_zs{}/clip_encoder_zs_{}_test/{}/{}/{}/{}_1.png".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method, idx))
        images[2].save("outputs_zs{}/clip_encoder_zs_{}_test/{}/{}/{}/{}_2.png".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method, idx))
        images[3].save("outputs_zs{}/clip_encoder_zs_{}_test/{}/{}/{}/{}_3.png".format(args.scale, neg_content_scale, args.content_prompt, args.target_content, args.method, idx))

    return images[0]


def main():
    args = parse_args()
    print(args.less_condition)
    ip_model = AutoIPAdapterZS(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["block"], less_condition=args.less_condition)

    # Load tokenizer
    text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(pipe.device, dtype=pipe.dtype)
    tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    # dataloader
    train_dataset = MyDataset(image_root_path=args.image_root_path, content_prompt=args.content_prompt)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    for i, batch in enumerate(train_dataloader):
        tokens = tokenizer(batch["texts"], return_tensors='pt').to(pipe.device)
        neg_content_embd = text_encoder(**tokens).text_embeds
        # neg_content_embd = text_encoder(batch["texts"]).text_embeds
        pil_image = model(batch["images"], neg_content_embd, batch["image_files"][0].split(".")[0], args, ip_model)


if __name__ == "__main__":
    main()











