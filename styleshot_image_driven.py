import os
from types import MethodType

import torch
import cv2
from annotator.hed import SOFT_HEDdetector
from annotator.lineart import LineartDetector
from diffusers import UNet2DConditionModel, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from huggingface_hub import snapshot_download
from ip_adapter_styleshot import StyleShot, StyleContentStableDiffusionControlNetPipeline
import argparse
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_root_path="style_evaluation_benchmark_v2",
                 content_prompt="person, animal, plant, or object in the foreground", tokenizer=None,):
        super().__init__()
        self.none_loop = 0
        self.tokenizer = tokenizer
        self.image_root_path = image_root_path
        self. content_prompt = content_prompt  # default setting: we do not need to give specific content description

        self.data, self.style = [], []
        all_styles = sorted(os.listdir(self.image_root_path))

        for style in all_styles:
            if style in all_styles[20:]:
                self.data.extend([os.path.join(image_root_path, style, fn) for fn in
                                  sorted(os.listdir(os.path.join(image_root_path, style)))])
                self.style.extend([style for _ in sorted(os.listdir(os.path.join(image_root_path, style)))])
        # print(self.data)
        # print(self.style)

    def load_data(self, item):
        try:
            text = item["content_prompt"]  # the content prompt need to be removed
        except Exception as e:
            print(e)
            text = ""
        image_file = item["image_file"]  # file name
        try:
            raw_image = Image.open(image_file)
        except Exception as e:
            print(e)
            raw_image = None
        return raw_image, text, image_file

    def __getitem__(self, idx):
        item = {"image_file": self.data[idx], "content_prompt": self.content_prompt}
        raw_image, text, image_file = self.load_data(item)
        image = raw_image.resize((512, 512))
        style = self.style[idx]

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image_file": image_file,
            "image": image,
            "text": text,
            "raw_image": raw_image,
            "content_text_input_ids": text_input_ids,
            "style": style,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = [example["image"] for example in data]  # torch.stack([example["image"] for example in data])
    texts = [example["text"] for example in data]
    styles = [example["style"] for example in data]
    image_files = [example["image_file"] for example in data]
    raw_images = [example["raw_image"].convert("RGB").resize((512, 512)) for example in data]
    text_input_ids = torch.cat([example["content_text_input_ids"] for example in data], dim=0)

    return {
        "image_files": image_files,
        "images": images,
        "texts": texts,
        "styles": styles,
        "raw_images": raw_images,
        "content_text_input_ids": text_input_ids,
    }


def main(args, batch, content, text_encoder, idx, prompt):
    style = batch["image_files"][0]
    style_image = Image.open(style)
    # processing content image
    content_image = cv2.imread(content)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    content_image = detector(content_image)
    content_image = Image.fromarray(content_image)

    # text_encoder = text_encoder.to(pipe.device, dtype=pipe.dtype)

    if args.method == "styleshot":
        neg_content_embd = None
    else:
        neg_content_embd = text_encoder(batch["content_text_input_ids"].to("cuda")).text_embeds

    print("check less_condition:", args.less_condition)
    
    generation = styleshot.generate(style_image=style_image, prompt=[[prompt]],
                                    content_image=content_image, seed=42, num_samples=4,
                                    neg_content_embd=neg_content_embd, less_condition=args.less_condition)

    os.makedirs(os.path.join(args.output, args.method + "_" + args.preprocessor, str(idx)), exist_ok=True)
    generation[0][0].save(os.path.join(args.output, args.method + "_" + args.preprocessor, str(idx),
                                       '{}_{}'.format(batch["styles"][0], "0_" + os.path.basename(batch["image_files"][0]))))
    generation[0][1].save(os.path.join(args.output, args.method + "_" + args.preprocessor, str(idx),
                                       '{}_{}'.format(batch["styles"][0], "1_" + os.path.basename(batch["image_files"][0]))))
    generation[0][2].save(os.path.join(args.output, args.method + "_" + args.preprocessor, str(idx),
                                       '{}_{}'.format(batch["styles"][0], "2_" + os.path.basename(batch["image_files"][0]))))
    generation[0][3].save(os.path.join(args.output, args.method + "_" + args.preprocessor, str(idx),
                                       '{}_{}'.format(batch["styles"][0], "3_" + os.path.basename(batch["image_files"][0]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="style_evaluation_benchmark_v2")
    parser.add_argument("--content", type=str, default="content")
    parser.add_argument("--preprocessor", type=str, default="Contour", choices=["Contour", "Lineart"])
    parser.add_argument("--prompt", type=str, default="text prompt")
    parser.add_argument("--output", type=str, default="outputs_image_driven")
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument(
        "--less_condition",
        default=False,
        action='store_true',
        help="Use image prompt adapter when less_condition is True, otherwise use text prompt adapter",
    )
    args = parser.parse_args()

    # dataloader
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

    content_prompt = "person, animal, plant, or object in the foreground"
    train_dataset = MyDataset(image_root_path=args.style, content_prompt=content_prompt, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
    )
    contents = []
    prompts = []
    for fn in [fn for fn in os.listdir(args.content)]:
        image_file = os.path.join(args.content, fn)
        contents.append(image_file)
        prompts.append(fn)

    base_model_path = "runwayml/stable-diffusion-v1-5"
    transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    device = "cuda"

    if args.preprocessor == "Lineart":
        detector = LineartDetector()
        styleshot_model_path = "Gaojunyao/StyleShot_lineart"
    elif args.preprocessor == "Contour":
        detector = SOFT_HEDdetector()
        styleshot_model_path = "Gaojunyao/StyleShot"
    else:
        raise ValueError("Invalid preprocessor")

    if not os.path.isdir(styleshot_model_path):
        styleshot_model_path = snapshot_download(styleshot_model_path, local_dir=styleshot_model_path)
        print(f"Downloaded model to {styleshot_model_path}")

    # weights for ip-adapter and our content-fusion encoder
    if not os.path.isdir(base_model_path):
        base_model_path = snapshot_download(base_model_path, local_dir=base_model_path)
        print(f"Downloaded model to {base_model_path}")
    if not os.path.isdir(transformer_block_path):
        transformer_block_path = snapshot_download(transformer_block_path, local_dir=transformer_block_path)
        print(f"Downloaded model to {transformer_block_path}")

    ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
    style_aware_encoder_path = os.path.join(styleshot_model_path, "pretrained_weight/style_aware_encoder.bin")

    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    content_fusion_encoder = ControlNetModel.from_unet(unet)

    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(base_model_path,
                                                                         controlnet=content_fusion_encoder)
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)

    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(pipe.device, dtype=pipe.dtype)

    # generate image
    for i, batch in enumerate(train_dataloader):
        print(batch["styles"][0])
        for idx, content in enumerate(contents):
            print(prompts[idx].split(".")[0])
            main(args, batch, content, text_encoder, idx, prompts[idx].split(".")[0])



