import os
import subprocess

os.environ["HF_HOME"] = "/playpen-nas-ssd/gongbang/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/playpen-nas-ssd/gongbang/.cache/huggingface/hub"

# target_ages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
img_path = "/playpen-nas-ssd/gongbang/benchmark_examples/inputs/thatcher/34_11_00.jpg"
input_age = 34
output_dir = "/playpen-nas-ssd/gongbang/exp_output/condition_inject/progression"
# output_dir = "/playpen-nas-ssd/gongbang/exp_output/condition_no_inject/progression"

# for target_age in target_ages:
#     cmd = f"""python edit.py  \
#         --source_prompt 'a photo of {input_age} year old person' \
#         --target_prompt 'a photo of {target_age} year old person' \
#         --guidance 2 \
#         --source_img_dir {img_path} \
#         --num_steps 30  \
#         --inject 5 \
#         --name 'flux-dev' \
#         --offload \
#         --output_dir {output_dir} """

#     subprocess.run(cmd, shell=True)
cmd = f"""python edit.py  \
    --source_prompt 'a photo of {input_age} year old person' \
    --target_prompt 'A portrait of 70-year-old woman with dry, uneven skin, deep wrinkles, and an unhealthy complexion. Her face looks saggy and dull, with dark circles under her eyes and age spots. Her hair is thinning or unkempt, and her posture appears slightly hunched' \
    --guidance 2 \
    --source_img_dir {img_path} \
    --num_steps 30  \
    --inject 5 \
    --name 'flux-dev' \
    --offload \
    --output_dir {output_dir} """

subprocess.run(cmd, shell=True)