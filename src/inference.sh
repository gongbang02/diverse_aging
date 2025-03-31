export HF_HOME="/playpen-nas-ssd/gongbang/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/playpen-nas-ssd/gongbang/.cache/huggingface/hub"

python edit.py  \
    --source_prompt 'A portrait of a 40-year-old woman' \
    --target_prompt 'A portrait of a 60-year-old woman with a slightly reddened complexion, visible broken capillaries around the nose and cheeks, and uneven skin tone. Fine lines and deeper wrinkles appear around her eyes and mouth, with mild puffiness in the eyelids and under-eye area. Her skin looks dehydrated, with a dull texture and slight sagging around the jawline. Dark circles and a tired expression give her a worn appearance.' \
    --guidance 2 \
    --source_img_dir '/playpen-nas-ssd/gongbang/project_experiments/trajectory/elizabeth/samecond/weight/img_40.jpg' \
    --num_steps 15  \
    --inject 3 \
    --name 'flux-dev' \
    --target_age 60 \
    --offload \
    --output_dir '/playpen-nas-ssd/gongbang/project_experiments/trajectory/elizabeth/samecond/weight/'