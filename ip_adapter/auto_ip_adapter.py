import os
from typing import List
from sklearn.cluster import KMeans
import numpy as np

import torch
from typing import Optional, Union, Any, Dict, Tuple, List, Callable
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.controlnet.pipeline_controlnet import retrieve_timesteps
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.models.controlnet import ControlNetModel
from torch import autograd
import torch.nn as nn
from torch.nn import functional as F
import math

from .utils import is_torch2_available, get_generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class FeatureAdapter_I(torch.nn.Module):
    """image (text) feature adapter model"""

    def __init__(self, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim)
        self.norm = torch.nn.LayerNorm(clip_embeddings_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_embeds = self.proj(embeds)
        clip_extra_context_embeds = self.norm(clip_extra_context_embeds)
        return clip_extra_context_embeds


class FeatureAdapter_T(torch.nn.Module):
    """image (text) feature adapter model"""

    def __init__(self, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Linear(768, clip_embeddings_dim)  # nn.Parameter(torch.ones([1, clip_embeddings_dim]))  #
        self.norm = torch.nn.LayerNorm(clip_embeddings_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_embeds = self.proj(embeds)  # embeds * self.proj  #
        clip_extra_context_embeds = self.norm(clip_extra_context_embeds)
        return clip_extra_context_embeds


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, target_blocks=["block"],
                 less_condition=True, content_feature_adapter=None):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.target_blocks = target_blocks

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        if less_condition:
            self.feature_adapter_model = self.init_adapter_I() if content_feature_adapter is None else content_feature_adapter.to(self.device)
        else:
            self.feature_adapter_model = self.init_adapter_T() if content_feature_adapter is None else content_feature_adapter.to(self.device)

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def init_adapter_I(self):
        feature_adapter_model = FeatureAdapter_I(
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
        ).to(self.device, dtype=torch.float16)
        return feature_adapter_model

    def init_adapter_T(self):
        feature_adapter_model = FeatureAdapter_T(
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
        ).to(self.device, dtype=torch.float16)
        return feature_adapter_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                selected = False
                for block_name in self.target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                    ).to(self.device, dtype=torch.float16)
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        skip=True
                    ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    # @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

        if content_prompt_embeds is not None:
            clip_image_embeds = clip_image_embeds - content_prompt_embeds

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            neg_content_emb=None,
            **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, content_prompt_embeds=neg_content_emb
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        # with torch.inference_mode():
        prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images

        
class AutoIPAdapter(IPAdapter):

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            neg_content_name=None,
            target_prompt_embeds=None,
            neg_content_scale=0.8,
            less_condition=True,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # with torch.inference_mode():
        prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        if neg_content_name is not None:
            auto_learner = AutoContentLearner(self.image_encoder, self.pipe.encode_prompt, neg_content_name, self.device,
                                               self.clip_image_processor, self.feature_adapter_model, num_samples, less_condition)

            pooled_prompt_embeds_ = auto_learner(pil_image)
            pooled_prompt_embeds_ *= neg_content_scale
            # print(pooled_prompt_embeds_)
        else:
            pooled_prompt_embeds_ = None
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image, content_prompt_embeds=pooled_prompt_embeds_,)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        # with torch.inference_mode():
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        #######################################################################################
        """
        pooled_prompt_embeds_ = self.image_proj_model(pooled_prompt_embeds_)  #
        pooled_prompt_embeds_ = pooled_prompt_embeds_.mean(1).repeat(1, 77, 1)
        prompt_embeds = torch.cat([pooled_prompt_embeds_, image_prompt_embeds], dim=1)  #
        # prompt_embeds = torch.cat([prompt_embeds, uncond_image_prompt_embeds], dim=1)
        """
        #######################################################################################
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
        

class AutoIPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            neg_content_name=None,
            neg_content_prompt=None,
            neg_content_scale=1.0,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # with torch.inference_mode():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        if neg_content_name is not None:
            # not used, need to modified and rewrite with AutoIPAdapterZSXL
            auto_learner = AutoContentLearner_(self.image_encoder, self.pipe.encode_prompt, neg_content_name,
                                              self.device, self.clip_image_processor, num_samples,
                                              self.feature_adapter_model, target_prompts=pooled_prompt_embeds,)
            pooled_prompt_embeds_ = auto_learner(pil_image)
            pooled_prompt_embeds_ *= neg_content_scale

        elif neg_content_prompt is not None:
            (
                prompt_embeds_,  # torch.Size([1, 77, 2048])
                negative_prompt_embeds_,
                pooled_prompt_embeds_,  # torch.Size([1, 1280])
                negative_pooled_prompt_embeds_,
            ) = self.pipe.encode_prompt(
                neg_content_prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            pooled_prompt_embeds_ *= neg_content_scale
        else:
            pooled_prompt_embeds_ = None

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image, content_prompt_embeds=pooled_prompt_embeds_)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        #######################################################################################
        """
        pooled_prompt_embeds_ = self.image_proj_model(pooled_prompt_embeds_)  #
        pooled_prompt_embeds_ = pooled_prompt_embeds_.mean(1).repeat(1, 77, 1)
        prompt_embeds = torch.cat([pooled_prompt_embeds_, image_prompt_embeds], dim=1)  #
        # prompt_embeds = torch.cat([prompt_embeds, uncond_image_prompt_embeds], dim=1)
        """
        #######################################################################################
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class AutoIPAdapterZSXL(IPAdapter):
    """ZeroShot Auto SDXL"""

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            neg_content_name=None,
            neg_content_embd=None,
            neg_content_prompt=None,
            neg_content_scale=1.0,
            theta=0.8,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # with torch.inference_mode():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        if neg_content_name is not None or neg_content_embd is not None:
            auto_learner = AutoContentLearnerZSXL(self.image_encoder, self.pipe.encode_prompt,
                                                neg_content_name, self.device,
                                                self.clip_image_processor, num_samples, theta, neg_content_scale)
            pooled_prompt_embeds_ = auto_learner(pil_image)
            pooled_prompt_embeds_ *= neg_content_scale

        elif neg_content_prompt is not None:
            (
                prompt_embeds_,  # torch.Size([1, 77, 2048])
                negative_prompt_embeds_,
                pooled_prompt_embeds_,  # torch.Size([1, 1280])
                negative_pooled_prompt_embeds_,
            ) = self.pipe.encode_prompt(
                neg_content_prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            pooled_prompt_embeds_ *= neg_content_scale
        else:
            pooled_prompt_embeds_ = None

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image, content_prompt_embeds=pooled_prompt_embeds_)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)


        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class AutoIPAdapterZS(IPAdapter):
    """ZeroShot Auto SD"""

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            guidance_scale=7.5,
            seed=None,
            num_inference_steps=30,
            content_image=None,
            neg_content_name=None,
            neg_content_embd=None,
            neg_content_prompt=None,
            neg_content_scale=1.0,
            theta=0.8,
            less_condition=False,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts


        # with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        if neg_content_embd is not None:
            print("check less less_condition", less_condition)
            auto_learner = AutoContentLearnerZS(self.image_encoder, self.pipe.encode_prompt,
                                                neg_content_embd, self.device,
                                                self.clip_image_processor, num_samples, theta,
                                                neg_content_scale, less_condition,)
            pooled_prompt_embeds_ = auto_learner(pil_image)
            pooled_prompt_embeds_ *= neg_content_scale

        elif neg_content_prompt is not None:
            (
                prompt_embeds_,  # torch.Size([1, 77, 2048])
                negative_prompt_embeds_,
                pooled_prompt_embeds_,  # torch.Size([1, 1280])
                negative_pooled_prompt_embeds_,
            ) = self.pipe.encode_prompt(
                neg_content_prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            pooled_prompt_embeds_ *= neg_content_scale
        else:
            pooled_prompt_embeds_ = None

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image, content_prompt_embeds=pooled_prompt_embeds_)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

        if content_image is None:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                **kwargs,
            ).images
        else:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                image=content_image,
                style_embeddings=image_prompt_embeds,
                negative_style_embeddings=uncond_image_prompt_embeds,
                **kwargs,
            ).images

        return images


# AutoContentLearner
class AutoContentLearner(nn.Module):
    def __init__(self, image_encoder, text_encoder, neg_content_name, device, clip_image_processor,
                 feature_adapter_model, num_samples, less_condition=True, target_prompt_embeds=None):
        super().__init__()
        self.image_encoder = image_encoder
        self.neg_content_name = neg_content_name
        self.device = device
        self.clip_image_processor = clip_image_processor
        self.less_condition = less_condition

        self.text_features = text_encoder(neg_content_name, device=self.device, num_images_per_prompt=1,
                                          do_classifier_free_guidance=True, )[0]

        if target_prompt_embeds is not None:
            self.target_prompt_embeds = target_prompt_embeds.clone()
        self.dtype = self.image_encoder.dtype
        # n_feat = self.image_encoder.config.projection_dim

        self.feature_adapter_model = feature_adapter_model

    def forward(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        image_feature = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds

        if self.less_condition:
            # extracting content feature from vanilla image feature
            adapted_feature = self.feature_adapter_model(image_feature)
        else:
            # extracting content feature from text feature
            adapted_feature = self.feature_adapter_model(self.text_features.mean(1).reshape(self.text_features.shape[0], -1))
            # adapted_feature = self.feature_adapter_model(self.text_features)

        s = image_feature.norm(dim=1) / adapted_feature.norm(dim=1)
        print(s.item())
        adapted_feature = s * adapted_feature

        return adapted_feature


# AutoContentLearner-ZeroShot
class AutoContentLearnerZSXL(nn.Module):
    def __init__(self, image_encoder, text_encoder, neg_content_name, device, clip_image_processor,
                 num_samples, theta, neg_content_scale, target_prompt_embeds=None):
        super().__init__()
        self.image_encoder = image_encoder
        self.neg_content_name = neg_content_name
        self.device = device
        self.clip_image_processor = clip_image_processor
        self.theta = theta
        self.neg_content_scale = neg_content_scale

        self.text_features = text_encoder(neg_content_name, device=self.device, num_images_per_prompt=num_samples,
                                          do_classifier_free_guidance=True, )[2]
        # self.text_features = self.text_features.mean(1).reshape(self.text_features.shape[0], -1)

        if target_prompt_embeds is not None:
            self.target_prompt_embeds = target_prompt_embeds.clone()

        self.dtype = self.image_encoder.dtype

    def forward(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        image_feature = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        image_feature_s = image_feature / image_feature.norm(dim=1, keepdim=True)
        text_features_s = self.text_features / self.text_features.norm(dim=1, keepdim=True)

        grad_zs = text_features_s.float()
        if self.theta < 0.0:
            data = (image_feature_s * grad_zs).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(data.detach().cpu().numpy())
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            mask1 = torch.tensor(labels == np.argmax(cluster_centers)).to(self.device)
            print(mask1.sum() / mask1.shape[-1])
        else:
            mask1 = ((image_feature_s * grad_zs) > torch.quantile((image_feature_s * grad_zs), self.theta, 1, True))

        adapted_feature = torch.where(mask1, image_feature, 0)

        return adapted_feature


# AutoContentLearner-ZeroShot
class AutoContentLearnerZS(nn.Module):
    def __init__(self, image_encoder, text_encoder, neg_content_embd, device, clip_image_processor,
                 num_samples, theta, neg_content_scale, less_condition, target_prompt_embeds=None):
        super().__init__()
        self.image_encoder = image_encoder

        self.device = device
        self.clip_image_processor = clip_image_processor
        self.theta = theta
        self.neg_content_scale = neg_content_scale

        self.text_features = neg_content_embd

        if target_prompt_embeds is not None:
            self.target_prompt_embeds = target_prompt_embeds.clone()

        self.dtype = self.image_encoder.dtype
        self.less_condition = less_condition
        print("check again:", self.less_condition)

    def forward(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        image_feature = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        image_feature_s = image_feature / image_feature.norm(dim=1, keepdim=True)
        text_features_s = self.text_features / self.text_features.norm(dim=1, keepdim=True)

        grad_zs = text_features_s.float()

        if self.less_condition:
            if self.theta < 0 and self.theta != -100.0:
                # ################# threshold method based on the pre-defined theta ##########
                mask1 = ((image_feature_s * grad_zs) < torch.quantile((image_feature_s * grad_zs), -self.theta, 1, True))
                adapted_feature = torch.where(mask1, image_feature, 0)
                # #############################################################################
            elif self.theta == -100.0:
                # """
                # # ################# Kmeans method: determine theta automatically ##########
                image_feature_s = image_feature 
                text_features_s = self.text_features
                data = (image_feature_s * grad_zs).reshape(-1, 1)
                kmeans = KMeans(n_clusters=2)
                kmeans.fit(data.detach().cpu().numpy())
                cluster_centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                mask1 = torch.tensor(labels == np.argmax(cluster_centers)).to(self.device)
                print(mask1.sum()/mask1.shape[-1])
                adapted_feature = torch.where(mask1, image_feature, 0)
                # # #############################################################################
                """
                s = (image_feature_s.float()) @ grad_zs.t()
                print("s:", s.item())
                x_list = (image_feature_s * grad_zs).detach().cpu().numpy()[0].tolist()
                x_list = sorted(x_list, reverse=True)
                cum_list = [np.sum(x_list[:i+1]) for i in range(len(x_list))]
                # print(cum_list)

                def find_nearest(array, value):
                    array = np.asarray(array)
                    for idx in range(len(x_list)):
                        diff = np.abs(array[idx] - value)
                        if diff < 1e-3:
                            return idx + 1

                x_list_bool = [(x_list[i] > 1e-4 and np.abs(image_feature_s.float().detach().cpu().numpy()[0,i]) > 1e-4
                                and np.abs(grad_zs.detach().cpu().numpy()[0,i] > 1e-4)) for i in range(len(x_list))]

                auto_theta = 1 - min([find_nearest(cum_list, s.item())/len(cum_list), np.sum(1 * x_list_bool)/len(cum_list)])
                auto_theta = auto_theta if s.item() > 0.001 else 1.0
                print("auto_theta:", auto_theta, find_nearest(cum_list, s.item()), np.sum(1 * x_list_bool))

                mask1 = ((image_feature_s * grad_zs) > torch.quantile((image_feature_s * grad_zs), auto_theta, 1, True))
                adapted_feature = torch.where(mask1, image_feature, 0)
                """
            else:
                # ################# threshold method based on the pre-defined theta ##########
                # print(self.theta)
                mask1 = ((image_feature_s * grad_zs) > torch.quantile((image_feature_s * grad_zs), self.theta, 1, True))
                adapted_feature = torch.where(mask1, image_feature, 0)
                # #############################################################################
        else:
            print(self.less_condition)
            adapted_feature = self.text_features

        # if self.target_prompt_embeds is not None:
        #     # m1:
        #     grad_zs = self.target_prompts.float()
        #     f = (image_feature - self.neg_content_scale * prompts_hg1).float()
        #     mask2 = (f * grad_zs.abs() > torch.quantile(f * grad_zs.abs(), theta, 1, True))
        #     prompts_hg = torch.where(mask2, image_feature - self.neg_content_scale * prompts_hg1, 0)
        #
        #     # m2:
        #     # grad_zs = self.target_prompts
        #     # feat1 = (image_feature - self.neg_content_scale * prompts_hg1) / (
        #     #         1e-6 + (image_feature - self.neg_content_scale * prompts_hg1).norm(dim=-1, keepdim=True))
        #     # feat2 = self.target_prompts / (1e-6 + self.target_prompts.norm(dim=-1, keepdim=True))
        #     # eta = feat1 @ feat2.t()
        #     # prompts_hg = eta * grad_zs
        #     return adapted_feature + prompts_hg

        return adapted_feature



class StyleContentStableDiffusionControlNetPipeline(StableDiffusionControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        style_embeddings: Optional[torch.FloatTensor] = None,
        negative_style_embeddings: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet. When `prompt` is a list, and if a list of images is passed for a single ControlNet,
                each will be paired with each prompt in the `prompt` list. This also applies to multiple ControlNets,
                where a list of image lists can be passed to batch for each prompt and each ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                if `do_classifier_free_guidance` is set to `True`.
                If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]

        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            # Nested lists as ControlNet condition
            if isinstance(image[0], list):
                # Transpose the nested image list
                image = [list(t) for t in zip(*image)]

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                if self.do_classifier_free_guidance:
                    style_embeddings_input = torch.cat([negative_style_embeddings, style_embeddings])

                # print(control_model_input.dtype, style_embeddings_input.dtype, image.dtype, t.dtype)
                self.controlnet.to(dtype=style_embeddings_input.dtype)
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=style_embeddings_input,
                    controlnet_cond=image.to(dtype=style_embeddings_input.dtype),
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)



