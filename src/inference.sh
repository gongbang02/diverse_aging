python edit.py  \
    --source_prompt "a photo of a 39 year old person" \
    --target_prompt "a photo of a x year old person" \
    --guidance 2 \
    --source_img_dir "/playpen-nas-ssd/gongbang/benchmark_examples/al/39_2.jpeg" \
    --num_steps 30  \
    --inject 5 \
    --name 'flux-dev' \
    --offload \
    --output_dir "/playpen-nas-ssd/gongbang/benchmark_examples/rf-solver-edit/al"