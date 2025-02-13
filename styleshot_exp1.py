import os
from types import MethodType

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from huggingface_hub import snapshot_download
from ip_adapter_styleshot import StyleShot
import argparse
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_root_path="style_evaluation_benchmark_v2",
                 content_prompt="person, animal, plant, or object in the foreground", tokenizer=None,):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_root_path = image_root_path
        self.content_prompt = content_prompt

        self.data, self.style = [], []

        self.data.extend([os.path.join(image_root_path, content_prompt, content_prompt, fn) for fn in
                          sorted(os.listdir(os.path.join(image_root_path, content_prompt, content_prompt)))])
        self.style.extend([_.split(".")[0] for _ in sorted(os.listdir(os.path.join(image_root_path, content_prompt, content_prompt)))])
        print(self.data)
        print(self.style)

    def load_data(self, item):
        try:
            text = "A {}".format(item["content_prompt"]) if item["content_prompt"] not in ["airplane", "automobile"] \
                else "An {}".format(item["content_prompt"])  # the content prompt need to be removed
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


def main(args, prompt, batch, text_encoder):
    if args.method == "styleshot":
        neg_content_embd = None
    else:
        neg_content_embd = text_encoder(batch["content_text_input_ids"].to("cuda")).text_embeds

    image = batch["images"][0]

    generation = styleshot.generate(style_image=image, prompt=[[prompt]], num_samples=1,
                                    neg_content_embd=neg_content_embd, less_condition=args.less_condition)

    os.makedirs(os.path.join(args.output, args.method, prompt.split(" ")[-1]), exist_ok=True)

    generation[0][0].save(os.path.join(args.output, args.method, prompt.split(" ")[-1], os.path.basename(batch["image_files"][0])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="data/object_test")
    parser.add_argument("--prompt", type=str, default="A wolf walking stealthily through the forest",)
    parser.add_argument("--output", type=str, default="outputs_exp1")
    parser.add_argument("--method", type=str, default="ours")
    parser.add_argument(
        "--less_condition",
        default=False,
        action='store_true',
        help="Use image prompt adapter when less_condition is True, otherwise use text prompt adapter",
    )
    args = parser.parse_args()

    prompts = ["A bench", "A human", "A laptop", "A rocket", "A flower"]
    src_contents = [ "dog", "cat", "frog", "automobile", "airplane", "horse", "ship", "truck", "deer", "bird"]

    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

    # dataloader
    for content_prompt in src_contents:
        train_dataset = MyDataset(image_root_path=args.style, content_prompt=content_prompt, tokenizer=tokenizer)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1,
        )

        # load model
        base_model_path = "runwayml/stable-diffusion-v1-5"
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
        styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(pipe.device, dtype=pipe.dtype)

        # generate image
        for i, batch in enumerate(train_dataloader):
            print(batch["styles"][0])
            for idx, prompt in enumerate(prompts):
                main(args, prompt, batch, text_encoder)
