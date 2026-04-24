# Modified to include DDPM_SRDE
import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger('base')


# =========================================================
# INITIALIZATION
# =========================================================

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()

    elif 'Linear' in classname:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()

    elif 'BatchNorm2d' in classname:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()

    elif 'Linear' in classname:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    logger.info(f'Initialization method [{init_type}]')

    if init_type == 'normal':
        net.apply(functools.partial(weights_init_normal, std=std))
    elif init_type == 'kaiming':
        net.apply(functools.partial(weights_init_kaiming, scale=scale))
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(f'Init {init_type} not implemented')


# =========================================================
# MODEL REGISTRY
# =========================================================

MODEL_REGISTRY = {
    "sr3": "sr3_modules",
    "ddpm": "ddpm_modules",
    "hdbmie": "hdbmie_modules",
}


# =========================================================
# MAIN FACTORY FUNCTION
# =========================================================

def define_G(opt):

    model_opt = opt['model']
    model_type = model_opt['which_model_G']

    if model_type not in MODEL_REGISTRY:
        raise NotImplementedError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    module_name = MODEL_REGISTRY[model_type]

    # ---------------- dynamic import ----------------
    if module_name == "sr3_modules":
        from .sr3_modules import diffusion, unet
    elif module_name == "ddpm_modules":
        from .ddpm_modules import diffusion, unet
    elif module_name == "hdbmie_modules":
        from .hdbmie_modules import diffusion, unet
    else:
        raise ImportError(module_name)


    # =========================================================
    # SAFE DEFAULTS
    # =========================================================
    model_opt['unet'].setdefault('norm_groups', 32)


    # =========================================================
    # BUILD UNET
    # =========================================================
    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        inner_channel=model_opt['unet']['inner_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )


    # =========================================================
    # WRAP DIFFUSION MODEL
    # =========================================================
    netG = diffusion.GaussianDiffusion(
        model=model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',
        conditional=model_opt['diffusion']['conditional'],
        timesteps=model_opt['diffusion'].get('timesteps', 1000)
    )


    # =========================================================
    # INIT WEIGHTS
    # =========================================================
    if opt['phase'] == 'train':
        init_weights(netG, init_type='orthogonal')


    # =========================================================
    # DISTRIBUTED WRAP
    # =========================================================
    if opt.get('gpu_ids') and opt.get('distributed'):
        netG = nn.DataParallel(netG)

    return netG
