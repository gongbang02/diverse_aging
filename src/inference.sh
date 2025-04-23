#!/bin/bash

export HF_HOME="/playpen-nas-ssd/gongbang/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/playpen-nas-ssd/gongbang/.cache/huggingface/hub"

python edit.py \
    --source_prompt 'a photo of person at 37 years old' \
    --target_prompt 'a photo of man at TARGETAGE years old' \
    --input_age 37 \
    --guidance 2 \
    --source_img_dir /playpen-nas-ssd3/gongbang/test_in/al/37_11.jpeg \
    --num_steps 15 \
    --inject 3 \
    --name 'flux-dev' \
    --offload \
    --output_dir /playpen-nas-ssd3/gongbang/age_test/try_k_strength0.3/al/37_11 \
    --person 'al'