import numpy as np
import os
import torch
from tqdm import tqdm
import argparse

import clip
from PIL import Image
import pandas as pd

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
# image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)

# model, preprocess = clip.load("ViT-L/14", device=device)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_path="outputs_exp2", method="ours"):
        super().__init__()
        self.root_path = root_path
        self.image, self.text, self.style, self.file_name = [], [], [], []
        self.method =method
        self.prompts = [
            "A bench", "A bird", "A butterfly", "An elephant", "A car", "A dog",
            "A cat", "A laptop", "A moose", "A penguin", "A robot", "A rocket",
            "An ancient temple surrounded by lush vegetation",
            "A chef preparing meals in kitchen",
            "A colorful butterfly resting on a flower",
            "A house with a tree beside",
            "A person jogging along a scenic trail",
            "A student walking to school with backpack",
            "A wolf walking stealthily through the forest",
            "A wooden sailboat docked in a harbor",
        ]

        self.fetch_data()

    def fetch_data(self):
        targets = os.listdir(os.path.join(self.root_path, method))
        for object_name in targets:
            prompt = self.prompts[int(object_name)]
            text = clip.tokenize([prompt])
            text = model.encode_text(text.to(device))  # text_encoder(text.to(device)).text_embeds  #
            image_names = sorted(os.listdir(os.path.join(self.root_path, self.method, object_name)))

            for i in tqdm(range(len(image_names))):
                image_name = image_names[i]
                image_path = os.path.join(self.root_path, self.method, object_name, image_name)
                image = preprocess(Image.open(image_path)).unsqueeze(0)

                style_name = image_name.replace("_" + image_name.split("_")[-1], "")#[:-2]
                style_file = image_name.split("_")[-1]
                data_path = "style_evaluation_benchmark_v2"
                style_path = os.path.join(data_path, style_name, style_file)
                style = preprocess(Image.open(style_path)).unsqueeze(0)
                self.image.append(image)
                self.style.append(style)
                self.text.append(text)
                self.file_name.append(style_name)
        self.image = torch.cat(self.image, 0)
        self.text = torch.cat(self.text, 0)
        self.style = torch.cat(self.style, 0)
        print(self.image.shape, self.text.shape, self.style.shape)
        return self.image, self.text, self.style, self.file_name

    def __getitem__(self, idx):
        image, text, style, file_name = self.image[idx], self.text[idx], self.style[idx], self.file_name[idx]

        return {
            "images": image,
            "texts": text,
            "styles": style,
            "file_names": file_name
        }

    def __len__(self):
        print(len(self.image))
        return len(self.image)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--method",
        type=str,
        default="ours",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="outputs_exp2",
        help="Training data root path",
    )
    parser.add_argument(
        "--less_condition",
        default=False,
        action='store_true',
        help="Use image prompt adapter when less_condition is True, otherwise use text prompt adapter",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    root_path = args.root_path
    results1 = {}
    results2 = {}
    method = args.method
    dataset = MyDataset(root_path=args.root_path, method=method)
    print("data_loaded.......")
    dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            # collate_fn=collate_fn,
            batch_size=100,
            num_workers=0,
        )
    for i, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        texts = batch["texts"].to(device)
        styles = batch["styles"].to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)  # image_encoder(images).image_embeds  #
            text_features = texts  # model.encode_text(texts)
            style_features = model.encode_image(styles)  # image_encoder(styles).image_embeds  #

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            style_features /= style_features.norm(dim=-1, keepdim=True)

            score_style = torch.diag(image_features @ style_features.t()).cpu().numpy().tolist()
            score_text = torch.diag(image_features @ text_features.t()).cpu().numpy().tolist()

            for i, file_name in enumerate(batch["file_names"]):
                if file_name not in results1.keys():
                    results1[file_name] = []
                if file_name not in results2.keys():
                    results2[file_name] = []
                results1[file_name].append(score_style[i])
                results2[file_name].append(score_text[i])

    # collect results to Excel file
    for key in results1.keys():
        results1[key] = np.mean(results1[key])
        print(key, results1[key])

    output_file = "v2_{}_style.xlsx".format(method)
    df = pd.DataFrame(results1, index=[0])
    df.to_excel(output_file, index=True)

    for key in results2.keys():
        results2[key] = np.mean(results2[key])
        print(key, results2[key])

    output_file = "v2_{}_text.xlsx".format(method)
    df = pd.DataFrame(results2, index=[0])
    df.to_excel(output_file, index=True)







