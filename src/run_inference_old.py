import os
import subprocess

os.environ["HF_HOME"] = "/playpen-nas-ssd/gongbang/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/playpen-nas-ssd/gongbang/.cache/huggingface/hub"

target_ages = [
    30,
    40, 
    50, 
    60, 
    70, 
    80, 
    90
    ]
# img_path = "/playpen-nas-ssd/gongbang/benchmark_examples/inputs/thatcher/34_11_00.jpg"
img_path = "/playpen-nas-ssd/gongbang/benchmark_examples/inputs/charles/20_13.jpeg"
# img_path = "/playpen-nas-ssd/gongbang/project_experiments/trajectory/thatcher/same_cond/alcoholism/img_70.jpg"
input_age = 20
# output_dir = "/playpen-nas-ssd/gongbang/exp_output/inverse/projv/thatcher"
# output_dir = "/playpen-nas-ssd/gongbang/exp_output/trajectory/alpha_vis/elizabeth/weight/"
output_dir = "/playpen-nas-ssd/gongbang/project_experiments/trajectory/charles/same_in/alcoholism/"

for target_age in target_ages:
    cmd = f"""python edit.py  \
        --source_prompt 'A portrait of a {input_age}-year-old man' \
        --target_prompt 'A portrait of a {target_age}-year-old man with a slightly reddened complexion, visible broken capillaries around his nose and cheeks, and an uneven skin tone. His face appears slightly puffy, especially around the eyes and jawline, with dark circles and mild under-eye bags. Fine lines on his forehead and around his mouth are more pronounced, and his skin looks dehydrated, dull, and slightly sagging. His lips may appear dry, and his overall expression seems fatigued and worn' \
        --target_age {target_age} \
        --guidance 2 \
        --source_img_dir {img_path} \
        --num_steps 15  \
        --inject 3 \
        --name 'flux-dev' \
        --offload \
        --output_dir {output_dir} """

    subprocess.run(cmd, shell=True)