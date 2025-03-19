import os
import subprocess

os.environ["HF_HOME"] = "/playpen-nas-ssd/gongbang/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/playpen-nas-ssd/gongbang/.cache/huggingface/hub"

# target_ages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
img_path = "/playpen-nas-ssd/gongbang/benchmark_examples/inputs/elizabeth/20_18.jpeg"
input_age = 20
output_dir = "/playpen-nas-ssd/gongbang/exp_output/condition_inject/progression/elizabeth"
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
    --target_prompt 'A 70-year-old woman with a red, bloated face and puffiness around the eyes, showing visible signs of alcohol consumption. Her skin looks dehydrated, with deep wrinkles, age spots, and uneven tone. Her hair is thinning or unkempt, and her posture is slightly hunched. She has a tired, worn expression' \
    --guidance 2 \
    --source_img_dir {img_path} \
    --num_steps 8  \
    --inject 1 \
    --start_layer_index 0 \
    --end_layer_index 37 \
    --name 'flux-dev' \
    --offload \
    --sampling_strategy 'rf_solver' \
    --output_prefix 'rf_solver' \
    --editing_strategy 'project_v' \
    --output_dir {output_dir} """

subprocess.run(cmd, shell=True)