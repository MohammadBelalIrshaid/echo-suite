import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import copy

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.ARCH
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ViFi_CLIP',
                      "vision_depth": cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.ViFi_CLIP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.ViFi_CLIP.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger, type = 'yes'):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = cfg.TRAINER.ViFi_CLIP.PROMPT_MODEL
        if type == 'yes':
            ctx_init = cfg.TRAINER.ViFi_CLIP.CTX_INIT
        else:
            ctx_init = cfg.TRAINER.ViFi_CLIP.CTX_INIT_NO
        ZS_evaluation = cfg.TRAINER.ViFi_CLIP.ZS_EVAL
        if ZS_evaluation:
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for c in classnames])
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            assert cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT >= 1, "In VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
            n_ctx = cfg.TRAINER.ViFi_CLIP.N_CTX_TEXT
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            logger.info(f"V-L design")
            logger.info(f'Initial text context: "{prompt_prefix}"')
            logger.info(f"Number of context words (tokens) for Language prompting: {n_ctx}")
            logger.info(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.ViFi_CLIP.N_CTX_VISION}")
            self.ctx = nn.Parameter(ctx_vectors)

            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
            # No prompting
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        if self.use_prompt_stage:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = self.complete_text_embeddings

        return prompts


class ViFiCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        self.M = 512
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        
        width = 768
        scale = width ** -0.5
        input_resolution = 224
        patch_size = 16
        output_dim = 512
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.image_encoder = clip_model.visual

        modulesno = list(self.image_encoder.transformer.resblocks.children())[8:]
        self.image_encoder_no = copy.deepcopy(nn.Sequential(*modulesno))
        self.image_encoder_inpost_no = copy.deepcopy(self.image_encoder.ln_post)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.image_attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.image_attention_no = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.projector = nn.Linear(1024, 512)
        self.projector_no = nn.Linear(1024, 512)

        self.projector_sqc = nn.Linear(1040, 512)

        self.classifier_sqc = nn.Linear(512, 3)

    def forward(self, image):
        # b = image.shape[0]
        # Lets encode the video into required format
        b, t, c, h, w = image.size()
        # Remove the batch dimensions
        image = image.reshape(-1, c, h, w)
        # Now pass the image into CLIP visual encoder
        image_features = self.image_encoder(image.type(self.dtype))

        image_features_no = self.image_encoder.conv1(image.type(self.dtype))
        image_features_no = image_features_no.reshape(image_features_no.shape[0], image_features_no.shape[1], -1)
        image_features_no = image_features_no.permute(0, 2, 1)

        image_features_no = torch.cat(
            [self.class_embedding.to(image_features_no.dtype) + torch.zeros(image_features_no.shape[0], 1, image_features_no.shape[-1], dtype=image_features_no.dtype, device=image_features_no.device),
             image_features_no], dim=1)
        image_features_no = image_features_no + self.positional_embedding.to(image_features_no.dtype)

        image_features_no = self.image_encoder.ln_pre(image_features_no)
        image_features_no = image_features_no.permute(1, 0, 2)

        modules = list(self.image_encoder.transformer.resblocks.children())[:8]
        share_net = nn.Sequential(*modules)
        image_features_no = share_net(image_features_no)    
        image_features_no = self.image_encoder_no(image_features_no)
        image_features_no = image_features_no.permute(1, 0, 2)
        image_features_no = self.image_encoder_inpost_no(image_features_no[:, 0, :])
        image_features_no = image_features_no @ self.proj

        # Now again attach the batch dimensions
        image_features = image_features.view(b, t, -1)  # [B, T, 512]
        image_features_no = image_features_no.view(b, t, -1)  # [B, T, 512]
        # Now take the mean along the temporal direction

        image_features1 = image_features.mean(dim=1, keepdim=False)  # image features are now ready  # [B, 512]
        image_features1_no = image_features_no.mean(dim=1, keepdim=False)  # image features are now ready # [B, 512]

        # # Attention
        attention_weights = self.image_attention(image_features)  # [B, T, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # softmax over T

        image_features2 = (image_features * attention_weights).sum(dim=1)  # [B, 512]
        
        attention_weights_no = self.image_attention_no(image_features_no)  # [B, T, 1]
        attention_weights_no = F.softmax(attention_weights_no, dim=1)  # softmax over T

        image_features2_no = (image_features * attention_weights_no).sum(dim=1)  # [B, 512]

        # # Concatenate the two features
        image_features = torch.cat([image_features1, image_features2], dim=1)  # [B, 1024]
        image_features_no = torch.cat([image_features1_no, image_features2_no], dim=1)


        weights_dis = F.softmax(attention_weights_no - attention_weights, dim=1).view(b, -1) # softmax over T

        image_features = self.projector(image_features)
        image_features_no = self.projector_no(image_features_no)
        image_features_sqc = self.projector_sqc(torch.cat([image_features, image_features_no, weights_dis], dim=1)) # [B, 512]

        logits_sqc = self.classifier_sqc(image_features_sqc)

        return logits_sqc

def returnCLIP(config, logger=None,
               class_names=None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    logger.info("Building ViFi-CLIP CLIP")
    model = ViFiCLIP(config, class_names, clip_model, logger)

    if config.TRAINER.ViFi_CLIP.PROMPT_MODEL:
        logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
    else:
        # Now need to control freezing of CLIP for fine-tuning
        train_complete_clip = config.TRAINER.ViFi_CLIP.USE
        if train_complete_clip == "both":
            logger.info("Turning on gradients for COMPLETE ViFi-CLIP model")
            for name, param in model.named_parameters():
                param.requires_grad_(True)
        else:
            if train_complete_clip == "image":
                logger.info("Turning on gradients for image side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "image_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            elif train_complete_clip == "no_text":
                logger.info("Turning on gradients for only no TEXT encoder side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "_no" in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            elif train_complete_clip == "quality":
                logger.info("Turning on gradients for quality side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "_sqc" in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            else:
                logger.info("Turning on gradients for TEXT side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "text_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Total learnable items: {len(enabled)}")
    model.float()
    return model
