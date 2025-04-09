import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline
from PIL import Image
import numpy as np

from diffusers import AutoPipelineForText2Image

import os
import math

NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

def get_target_ages(input_age):
    """Generate a list of target ages in multiples of 10, starting from the next multiple of 10 up to 90."""
    start = (math.floor(input_age / 10) + 1) * 10
    return list(range(start, 91, 10))

@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    offload: bool = False,
    add_sampling_metadata: bool = True,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    torch.set_grad_enabled(False)
    name = args.name
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt
    guidance = args.guidance
    output_dir = args.output_dir
    num_steps = args.num_steps
    offload = args.offload
    input_age = args.input_age

    target_ages = get_target_ages(input_age)

    # nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # Load LoRA weights if specified
    # if args.lora_path:
    #     # First load the base model
    #     pipe = AutoPipelineForText2Image.from_pretrained(
    #         "/playpen-nas-ssd/gongbang/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44", 
    #         torch_dtype=torch.bfloat16,
    #         use_peft=True
    #     )
    #     # Load LoRA weights
    #     pipe.load_lora_weights(
    #         args.lora_path, 
    #         weight_name='pytorch_lora_weights.safetensors'
    #     )
    #     # Extract the state dict
    #     model_state_dict = pipe.transformer.state_dict()
    #     del pipe  # Free memory
    # else:
    #     model_state_dict = None

    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    # # Apply LoRA weights if available
    # if model_state_dict is not None:
    #     model.load_state_dict(model_state_dict, strict=False)
    #     print('loaded lora for transformer')
    #     # for k, v in model_state_dict.items():
    #     #     if "lora" in k:
    #     #         print(k)
        
    #     # sanity check: Verify the model has LoRA weights
    #     has_lora = any("lora" in k for k in model_state_dict.keys())
    #     if not has_lora:
    #         print("Warning: No LoRA weights found in the loaded state dict!")
    

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)
    
    init_image = None
    init_image = np.array(Image.open(args.source_img_dir).convert('RGB'))
    
    shape = init_image.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image[:new_h, :new_w, :]

    width, height = init_image.shape[0], init_image.shape[1]
    init_image = encode(init_image, torch_device, ae)

    num_processed = 0
    info = {}
    info['feature_path'] = args.feature_path
    info['feature'] = {}
    info['inject_step'] = args.inject
    print(f"running edit on target ages: {target_ages}")
    for target_age in target_ages:
        target_age_prompt = target_prompt.replace("TARGETAGE", str(target_age))
        print(f"Target prompt: {target_age_prompt}")

        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_age_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if loop:
            opts = parse_prompt(opts)

        while opts is not None:
            if opts.seed is None:
                opts.seed = rng.seed()
            print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
            t0 = time.perf_counter()

            opts.seed = None
            if offload:
                ae = ae.cpu()
                torch.cuda.empty_cache()
                t5, clip = t5.to(torch_device), clip.to(torch_device)

            
            if not os.path.exists(args.feature_path):
                os.mkdir(args.feature_path)

            inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

            # offload TEs to CPU, load model to gpu
            if offload:
                t5, clip = t5.cpu(), clip.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)

            # if num_processed == 0:
            #     # inversion initial noise
            #     print("Doing inversion with input image")
            #     z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
            #     info['feature']["initial_noise"] = z
                
            # inp_target["img"] = info['feature']["initial_noise"]

            # inversion initial noise
            z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
            
            inp_target["img"] = z

            timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

            # if num_processed % 2 == 1:
            #     info['update_v'] = True
            #     print(f"Attempt to update feature to v after processing {num_processed} images")

            # denoise initial noise
            x, _ = denoise(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

            # num_processed += 1
            
            if offload:
                model.cpu()
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            # decode latents to pixel space
            batch_x = unpack(x.float(), opts.width, opts.height)

            for x in batch_x:
                x = x.unsqueeze(0)
                output_name = os.path.join(output_dir, f"img_{target_age}.jpg")
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                print(f"Done in {t1 - t0:.1f}s. Saving {output_name}")
                x = x.clamp(-1, 1)
                x = embed_watermark(x.float())
                x = rearrange(x[0], "c h w -> h w c")

                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                # nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
                
                # exif_data = Image.Exif()
                # exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                # exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                # exif_data[ExifTags.Base.Model] = name
                # if add_sampling_metadata:
                #     exif_data[ExifTags.Base.ImageDescription] = source_prompt
                img.save(output_name)

                if loop:
                    print("-" * 80)
                    opts = parse_prompt(opts)
                else:
                    opts = None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RF-Edit')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--source_img_dir', default='', type=str,
                        help='The path of the source image')
    parser.add_argument('--source_prompt', type=str,
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target_prompt', type=str,
                        help='describe the requirement of editing')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--guidance', type=float, default=5,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=25,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--inject', type=int, default=20,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='the path of the edited image')
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA weights directory')
    parser.add_argument('--input_age', type=int, default=30,
                        help='the input age of the source image')

    args = parser.parse_args()

    main(args)
