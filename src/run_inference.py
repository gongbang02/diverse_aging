import os
import subprocess
import math

# Set environment variables
os.environ["HF_HOME"] = "/playpen-nas-ssd/gongbang/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/playpen-nas-ssd/gongbang/.cache/huggingface/hub"

# Function to extract the input age from the filename (before the first underscore)
def extract_input_age(image_path):
    """Extract the input age from the filename (before the first underscore)."""
    filename = os.path.basename(image_path)
    age_str = filename.split('_')[0]
    return int(age_str)

# Function to generate a list of target ages
def get_target_ages(input_age):
    """Generate a list of target ages in multiples of 10, starting from the next multiple of 10 up to 90."""
    start = (math.floor(input_age / 10) + 1) * 10
    return list(range(start, 91, 10))

# Function to run the edit.py script with the given parameters
def run_edit(source_img, input_age, target_age, target_prompt, output_dir):
    """Run the edit.py script with the given parameters."""
    formatted_prompt = target_prompt.format(age=target_age)  # Replace {age} with target_age
    print(f"Target prompt {target_age}: {formatted_prompt}")
    cmd = (
        f"python edit.py "
        f"--source_prompt 'A portrait of a {input_age}-year-old man' "
        f"--target_prompt \"{formatted_prompt}\" "
        f"--target_age {target_age} "
        f"--guidance 2 "
        f"--source_img_dir {source_img} "
        f"--num_steps 15 "
        f"--inject 3 "
        f"--name 'flux-dev' "
        f"--offload "
        f"--output_dir {output_dir} "
    )
    print(f"Running command for target age {target_age}:")
    print(cmd)
    subprocess.run(cmd, shell=True)
    
    new_img = os.path.join(output_dir, f"img_{target_age}.jpg")
    return new_img

# Function to process a block of two target ages, updating the source image for the next iteration
def process_target_block(source_img, current_input_age, target_age_pair, target_prompts, output_dir):
    """Process a block of two target ages, updating the source image for the next iteration."""
    out_paths = []
    for target_age in target_age_pair:
        prompt_template = target_prompts.get(target_age, "A portrait of a {age}-year-old man with subtle aging features.")
        new_source = run_edit(source_img, current_input_age, target_age, prompt_template, output_dir)
        out_paths.append(new_source)
    return out_paths[-1], target_age_pair[-1]

# -------------------------
# Main processing starts here.
# -------------------------
men = ["al", "charles", "chow", "diego", "jackie", "robert"]
women = ["elaine", "elizabeth", "jennifer", "nicole", "thatcher", 'oprah']


for person in men:

    input_folder = f"/playpen-nas-ssd/gongbang/test_inputs/{person}"  # Change this to your input folder path
    output_root_dir = f"/playpen-nas-ssd/gongbang/exp_results/{person}/samecond/loseweight/"  # Change this to your output root folder

    # List all images in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    print(f"Found image files: {image_files}")

    # Define prompt templates with {age} placeholders
    woman_weight = "A portrait of a {age}-year-old woman with a fuller, rounder face, prominent cheeks, and a softer jawline. Her chin appears less defined, with subtle fat deposits around the neck. Her skin is smooth but slightly stretched, with faint nasolabial folds. Eyelids are slightly puffier, and her cheeks have a healthy, plump appearance."
    woman_hairloss = "A portrait of a {age}-year-old woman with noticeable hair thinning, particularly along the part and temples. Her hairline is slightly receding, revealing more of her forehead. Strands appear finer, with some sparse areas exposing the scalp. Her eyebrows may also appear slightly thinner, and her overall hair volume is reduced, making her facial contours more pronounced."
    woman_sunlight = "A portrait of a {age}-year-old woman with slightly tanned skin, fine lines around her eyes and mouth, and subtle sunspots across her cheeks and forehead. Her skin appears slightly rougher, with a faint leathery texture. Crow’s feet are more pronounced, and her lips have slight vertical lines. Her hair may have lighter, sun-bleached strands, and her overall complexion has a weathered yet warm glow."
    woman_poorskin = "A portrait of a {age}-year-old woman with uneven skin tone, visible pores, and a slightly rough texture. Fine lines on her forehead and around her mouth appear more pronounced, with mild sagging near the jawline. Her complexion looks dull, with occasional blemishes, redness, or dry patches. Dark circles or puffiness may be present under her eyes, and her skin has a slightly tired, lackluster appearance."
    woman_alcoholism = "A portrait of a {age}-year-old woman with a slightly reddened complexion, visible broken capillaries around the nose and cheeks, and uneven skin tone. Fine lines and deeper wrinkles appear around her eyes and mouth, with mild puffiness in the eyelids and under-eye area. Her skin looks dehydrated, with a dull texture and slight sagging around the jawline. Dark circles and a tired expression give her a worn appearance."
    woman_loseweight = "A portrait of a {age}-year-old woman with a lean, well-defined face, prominent cheekbones, and a more sculpted jawline. Her skin appears firm and slightly taut, with reduced fullness around her cheeks and under her chin. Fine lines around her eyes and mouth are subtly visible, but her complexion looks radiant and healthy. Her eyes appear more alert, and her overall facial structure is more toned, reflecting improved muscle definition and lower body fat."
    woman_goodskin = "A portrait of a {age}-year-old woman with smooth, even-toned skin, a radiant complexion, and a healthy glow. Fine lines around her eyes and mouth are minimal, with firm, well-hydrated skin that appears plump and elastic. Her pores are refined, and her under-eye area looks bright and refreshed, free of noticeable dark circles or puffiness. Her overall facial appearance is youthful, vibrant, and well-maintained."

    man_weight = "A portrait of a {age}-year-old man with a fuller, rounder face, softened jawline, and less-defined chin. His cheeks appear plumper, with subtle fat deposits around the neck. Nasolabial folds are slightly deeper, and his eyelids may look puffier. His skin appears smoother but slightly stretched, with a fuller under-chin area contributing to a softer facial contour."
    man_hairloss = "A portrait of a {age}-year-old man with a receding hairline and thinning hair on the crown. His forehead appears more prominent, with fine lines becoming more visible. The remaining hair is slightly finer, and sparse areas expose more of the scalp. His eyebrows may appear slightly thinner, and his facial features seem more defined due to reduced hair framing his face."
    man_sunlight = "A portrait of a {age}-year-old man with tanned, weathered skin, fine lines etched around his eyes and mouth, and deeper wrinkles on his forehead. His complexion has a slightly rough texture with visible sunspots on his cheeks and forehead. Crow’s feet are more pronounced, and his skin appears slightly leathery with mild sagging around the jawline. His hair may have subtle sun-bleached strands, and his lips show faint dryness or cracking."
    man_poorskin = "A portrait of a {age}-year-old man with uneven skin texture, enlarged pores, and a slightly rough, dull complexion. Fine lines are visible on his forehead and around his eyes, with mild sagging near the jawline. His skin appears dehydrated, with occasional redness, blemishes, or dry patches. Dark circles or puffiness under his eyes contribute to a tired appearance, and his overall complexion lacks vibrancy."
    man_alcoholism = "A portrait of a {age}-year-old man with a slightly reddened complexion, visible broken capillaries around his nose and cheeks, and an uneven skin tone. His face appears slightly puffy, especially around the eyes and jawline, with dark circles and mild under-eye bags. Fine lines on his forehead and around his mouth are more pronounced, and his skin looks dehydrated, dull, and slightly sagging. His lips may appear dry, and his overall expression seems fatigued and worn."
    man_loseweight = "A portrait of a {age}-year-old man with a lean, well-defined face, prominent cheekbones, and a sharper jawline. His skin appears firm and slightly taut, with reduced fullness around the cheeks and neck. Fine lines on his forehead and around his eyes are subtly visible, but his complexion looks healthier and more vibrant. His eyes appear more alert, and his overall facial structure is more chiseled, reflecting improved muscle tone and lower body fat."
    man_goodskin = "A portrait of a {age}-year-old man with smooth, even-toned skin, a well-hydrated complexion, and a healthy glow. Fine lines on his forehead and around his eyes are minimal, and his skin appears firm with good elasticity. His pores are refined, and his under-eye area looks refreshed without noticeable dark circles or puffiness. His overall facial appearance is vibrant, youthful, and well-maintained."


    target_prompts = {
        30: man_loseweight,
        40: man_loseweight,
        50: man_loseweight,
        60: man_loseweight,
        70: man_loseweight,
        80: man_loseweight,
        90: man_loseweight,
    }

    for img_filename in image_files:
        img_path = os.path.join(input_folder, img_filename)
        
        # Extract the input age from the image filename
        input_age = extract_input_age(img_path)
        print(f"Processing image: {img_filename}, Extracted input age: {input_age}")
        
        # Generate target ages based on the input age
        target_ages = get_target_ages(input_age)
        print(f"Target ages for {img_filename}: {target_ages}")
        
        # Create output directory based on the input image filename
        output_dir = os.path.join(output_root_dir, os.path.splitext(img_filename)[0])
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")
        
        # Split the target ages into blocks of 2
        blocks = [target_ages[i:i+2] for i in range(0, len(target_ages), 2)]
        print(f"Processing in blocks: {blocks}")

        # Process the target ages in blocks
        current_source = img_path
        current_age = input_age
        for block in blocks:
            print(f"\nProcessing block for target ages: {block}")
            current_source, current_age = process_target_block(current_source, current_age, block, target_prompts, output_dir)
            print(f"After processing block {block}, new input source is {current_source} and input age updated to {current_age}")

    print("Processing complete.")
