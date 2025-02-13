import numpy as np
import os
import torch
from tqdm import tqdm
import argparse

import clip
from PIL import Image
import pandas as pd

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_path="outputs_exp1", method="ours"):
        super().__init__()
        self.root_path = root_path
        self.image, self.text, self.style, self.file_name = [], [], [], []
        self.negiative_content_prompt = []
        self.negiative_content_prompt = []
        self.method =method
        self.fetch_data()

    def is_image_black(self, image_path): 
        with Image.open(image_path) as img: 
            for pixel in img.getdata(): 
                if pixel != (0,0,0): 
                    return False 
                else:
                    return True

    def fetch_data(self):
        objects = os.listdir(os.path.join(self.root_path, self.method))
        for object_name in objects:
            print(object_name)
            prompt = "An " + object_name if object_name in ["automobile, airplane"] else "A " + object_name
#            text = clip.tokenize([prompt])
#            text = model.encode_text(text.to(device))

            text = tokenizer(
                    prompt,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
            text = text_encoder(text.to(device)).text_embeds
            image_names = sorted(os.listdir(os.path.join(self.root_path, self.method, object_name)))

            for i in tqdm(range(len(image_names))):
                image_name = image_names[i]
                image_path = os.path.join(self.root_path, self.method, object_name, image_name)
                if self.is_image_black(image_path):
                    print("! all black figure")
                    continue
                image = preprocess(Image.open(image_path)).unsqueeze(0)

                negative_content = image_name.split("-")[1]
                style_name = image_name
                data_path = "data/object_test"
                style_path = os.path.join(data_path, "{}/{}".format(negative_content, negative_content), style_name)
                style = preprocess(Image.open(style_path)).unsqueeze(0)  ################

                style_prompt = (image_name.split("-")[-2] + " " + image_name.split("-")[-1].split("_")[0]).replace("of", "")
                style_prompt = clip.tokenize([style_prompt])
                self.image.append(image)
                self.style.append(style)  
                # self.style.append(style_prompt)
                self.text.append(text)
                self.file_name.append(negative_content + "_" + object_name)
                negative_content = "An " + negative_content if negative_content in ["automobile, airplane"] else "A " + negative_content
                # negative_content_prompt = clip.tokenize([negative_content])
                negative_content_prompt = tokenizer(
                        negative_content,
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids
                
                self.negiative_content_prompt.append(negative_content_prompt)
        self.image = torch.cat(self.image, 0)
        self.text = torch.cat(self.text, 0)
        self.style = torch.cat(self.style, 0)
        self.negiative_content_prompt = torch.cat(self.negiative_content_prompt, 0)
        # print(self.image.shape, self.text.shape, self.style.shape)
        return self.image, self.text, self.style, self.file_name, self.negiative_content_prompt


    def __getitem__(self, idx):
        image, text, style, file_name = self.image[idx], self.text[idx], self.style[idx], self.file_name[idx]
        negiative_content_prompt = self.negiative_content_prompt[idx]

        return {
            "images": image,
            "texts": text,
            "styles": style,
            "file_names": file_name,
            "negiative_content_prompts": negiative_content_prompt
        }

    def __len__(self):
        print(len(self.image))
        return len(self.image)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
            "--content_prompt",
            type=str,
            default="ship",
            help="The content in style reference.",
        )
    parser.add_argument(
        "--root_path",
        type=str,
        default="outputs_exp1",
        help="Training data root path",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ours",
    )
    parser.add_argument(
        "--less_condition",
        default=False,
        action='store_true',
        help="Use image prompt adapter when less_condition is True, otherwise use text prompt adapter",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.8,
        help="the ratio of masked elements in image feature.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # permutation result files
    root_path = args.root_path

    results1 = {}
    results2 = {}
    results3 = {}
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
        negiative_content_prompts = batch["negiative_content_prompts"].to(device)
        with torch.no_grad():
#            image_features = model.encode_image(images)
#            text_features = texts
#            style_features = model.encode_image(styles) 
#            # style_features = model.encode_text(styles)
#            negiative_content_features = model.encode_text(negiative_content_prompts)
#            # style_features -= negiative_content_features

            image_features = image_encoder(images).image_embeds  # model.encode_image(images)
            text_features = texts
            style_features = image_encoder(styles).image_embeds
            negiative_content_features = text_encoder(negiative_content_prompts.to(device)).text_embeds

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            style_features /= style_features.norm(dim=-1, keepdim=True)
            negiative_content_features /= negiative_content_features.norm(dim=-1, keepdim=True)
            style_features_ = style_features - negiative_content_features
            # text_features -= negiative_content_features

            score_style = torch.diag(image_features @ style_features_.t()).cpu().numpy().tolist()
            score_text = torch.diag(image_features @ text_features.t()).cpu().numpy().tolist()
            score_neg_text = torch.diag(image_features @ negiative_content_features.t()).cpu().numpy().tolist()

            scale = torch.diag(style_features @ negiative_content_features.t()).cpu().numpy().tolist()
            
            score_style = [score_style[i] if score_neg_text[i] <= score_text[i] else 0 for i in range(len(scale))]
            # score_text = [score_text[i] if score_neg_text[i] <= score_text[i] else -1 for i in range(len(scale))]
            score_text = [1 if score_neg_text[i] <= score_text[i] else 0 for i in range(len(scale))]
            score_neg_text = [score_neg_text[i] / scale[i] if score_neg_text[i] <= score_text[i] else 1 for i in range(len(scale))]
            

            for i, file_name in enumerate(batch["file_names"]):
                if file_name not in results1.keys():
                    results1[file_name] = []
                if file_name not in results2.keys():
                    results2[file_name] = []
                if file_name not in results3.keys():
                    results3[file_name] = []
                results1[file_name].append(score_style[i])
                results2[file_name].append(score_text[i])
                results3[file_name].append(score_neg_text[i])

    # collect results to Excel file
    for key in results1.keys():
        results1[key] = np.mean(results1[key])
        print(key, results1[key])

    output_file = "results/exp1_metric2/style_{}.xlsx".format(method)
    df = pd.DataFrame(results1, index=[0])
    df.to_excel(output_file, index=True)

    for key in results2.keys():
        results2[key] = np.mean(results2[key])
        print(key, results2[key])
    output_file = "results/exp1_metric2/text_{}.xlsx".format(method)
    df = pd.DataFrame(results2, index=[0])
    df.to_excel(output_file, index=True)

    for key in results3.keys():
        results3[key] = np.mean(results3[key])
        print(key, results3[key])
    output_file = "results/exp1_metric2/neg_text_{}.xlsx".format(method)
    df = pd.DataFrame(results3, index=[0])
    df.to_excel(output_file, index=True)







