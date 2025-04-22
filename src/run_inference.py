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


# Function to run the edit.py script with the given parameters
def run_edit(source_img, input_age, input_prompt, target_prompt, output_dir, person):
    cmd = (
        f"python edit.py "
        f"--source_prompt '{input_prompt}' "
        f"--target_prompt '{target_prompt}' "
        f"--input_age {input_age} "
        f"--guidance 2 "
        f"--source_img_dir {source_img} "
        f"--num_steps 15 "
        f"--inject 3 "
        f"--name 'flux-dev' "
        f"--offload "
        f"--output_dir {output_dir} "
        f"--person {person} "
    )
    subprocess.run(cmd, shell=True)


# -------------------------
# Main processing starts here.
# -------------------------
men = [
    "al", 
    "charles", 
    "chow", 
    "diego", 
    "jackie", 
    "robert",
    ]
women = [
    "elaine", 
    "elizabeth", 
    "jennifer", 
    "nicole", 
    "thatcher", 
    "oprah",
    ]

persons = [
    "al", 
    # "charles", 
    # "chow", 
    # "diego", 
    # "jackie", 
    # "robert",
    # "elaine", 
    # "elizabeth", 
    # "jennifer", 
    # "nicole", 
    # "thatcher", 
    # "oprah",
]

ethnicity = {
    "al": "white",
    "charles": "white",
    "chow": "asian", 
    "diego": "hispanic", 
    "jackie": "asian", 
    "robert": "white",
    "elaine": "asian",
    "elizabeth": "white",
    "jennifer": "white",
    "nicole": "white",
    "thatcher": "white",
    "oprah": "black",
}


for person in persons:

    input_folder = f"/playpen-nas-ssd/gongbang/test_inputs/{person}"
    output_root_dir = f"/playpen-nas-ssd3/gongbang/age_test/try_k_results/{person}/"  # Change this to your output root folder

    # List all images in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    print(f"Found image files: {image_files}")

    person_ethnicity = ethnicity[person]

    # Define prompt templates with age placeholders
    woman_weight = f"A portrait of a TARGETAGE-year-old {person_ethnicity} woman with a fuller, rounder face, prominent cheeks, and a softer jawline. Her chin appears less defined, with subtle fat deposits around the neck. Her skin is smooth but slightly stretched, with faint nasolabial folds. Eyelids are slightly puffier, and her cheeks have a healthy, plump appearance."
    woman_hairloss = f"A portrait of a TARGETAGE-year-old {person_ethnicity} woman with noticeable hair thinning, particularly along the part and temples. Her hairline is slightly receding, revealing more of her forehead. Strands appear finer, with some sparse areas exposing the scalp. Her eyebrows may also appear slightly thinner, and her overall hair volume is reduced, making her facial contours more pronounced."
    woman_sunlight = f"A portrait of a TARGETAGE-year-old {person_ethnicity} woman with slightly tanned skin, fine lines around her eyes and mouth, and subtle sunspots across her cheeks and forehead. Her skin appears slightly rougher, with a faint leathery texture. Crow’s feet are more pronounced, and her lips have slight vertical lines. Her hair may have lighter, sun-bleached strands, and her overall complexion has a weathered yet warm glow."
    woman_poorskin = f"A portrait of a TARGETAGE-year-old {person_ethnicity} woman with uneven skin tone, visible pores, and a slightly rough texture. Fine lines on her forehead and around her mouth appear more pronounced, with mild sagging near the jawline. Her complexion looks dull, with occasional blemishes, redness, or dry patches. Dark circles or puffiness may be present under her eyes, and her skin has a slightly tired, lackluster appearance."
    woman_alcoholism = f"A portrait of a TARGETAGE-year-old {person_ethnicity} woman with a slightly reddened complexion, visible broken capillaries around the nose and cheeks, and uneven skin tone. Fine lines and deeper wrinkles appear around her eyes and mouth, with mild puffiness in the eyelids and under-eye area. Her skin looks dehydrated, with a dull texture and slight sagging around the jawline. Dark circles and a tired expression give her a worn appearance."
    woman_loseweight = f"A portrait of a TARGETAGE-year-old {person_ethnicity} woman with a lean, well-defined face, prominent cheekbones, and a more sculpted jawline. Her skin appears firm and slightly taut, with reduced fullness around her cheeks and under her chin. Fine lines around her eyes and mouth are subtly visible, but her complexion looks radiant and healthy. Her eyes appear more alert, and her overall facial structure is more toned, reflecting improved muscle definition and lower body fat."
    woman_goodskin = f"A portrait of a TARGETAGE-year-old {person_ethnicity} woman with smooth, even-toned skin, a radiant complexion, and a healthy glow. Fine lines around her eyes and mouth are minimal, with firm, well-hydrated skin that appears plump and elastic. Her pores are refined, and her under-eye area looks bright and refreshed, free of noticeable dark circles or puffiness. Her overall facial appearance is youthful, vibrant, and well-maintained."
    woman_age = f"a photo of person at TARGETAGE years old"

    man_weight = f"A portrait of a TARGETAGE-year-old {person_ethnicity} man with a fuller, rounder face, softened jawline, and less-defined chin. His cheeks appear plumper, with subtle fat deposits around the neck. Nasolabial folds are slightly deeper, and his eyelids may look puffier. His skin appears smoother but slightly stretched, with a fuller under-chin area contributing to a softer facial contour."
    man_hairloss = f"A portrait of a TARGETAGE-year-old {person_ethnicity} man with a receding hairline and thinning hair on the crown. His forehead appears more prominent, with fine lines becoming more visible. The remaining hair is slightly finer, and sparse areas expose more of the scalp. His eyebrows may appear slightly thinner, and his facial features seem more defined due to reduced hair framing his face."
    man_sunlight = f"A portrait of a TARGETAGE-year-old {person_ethnicity} man with tanned, weathered skin, fine lines etched around his eyes and mouth, and deeper wrinkles on his forehead. His complexion has a slightly rough texture with visible sunspots on his cheeks and forehead. Crow’s feet are more pronounced, and his skin appears slightly leathery with mild sagging around the jawline. His hair may have subtle sun-bleached strands, and his lips show faint dryness or cracking."
    man_poorskin = f"A portrait of a TARGETAGE-year-old {person_ethnicity} man with uneven skin texture, enlarged pores, and a slightly rough, dull complexion. Fine lines are visible on his forehead and around his eyes, with mild sagging near the jawline. His skin appears dehydrated, with occasional redness, blemishes, or dry patches. Dark circles or puffiness under his eyes contribute to a tired appearance, and his overall complexion lacks vibrancy."
    man_alcoholism = f"A portrait of a TARGETAGE-year-old {person_ethnicity} man with a slightly reddened complexion, visible broken capillaries around his nose and cheeks, and an uneven skin tone. His face appears slightly puffy, especially around the eyes and jawline, with dark circles and mild under-eye bags. Fine lines on his forehead and around his mouth are more pronounced, and his skin looks dehydrated, dull, and slightly sagging. His lips may appear dry, and his overall expression seems fatigued and worn."
    man_loseweight = f"A portrait of a TARGETAGE-year-old {person_ethnicity} man with a lean, well-defined face, prominent cheekbones, and a sharper jawline. His skin appears firm and slightly taut, with reduced fullness around the cheeks and neck. Fine lines on his forehead and around his eyes are subtly visible, but his complexion looks healthier and more vibrant. His eyes appear more alert, and his overall facial structure is more chiseled, reflecting improved muscle tone and lower body fat."
    man_goodskin = f"A portrait of a TARGETAGE-year-old {person_ethnicity} man with smooth, even-toned skin, a well-hydrated complexion, and a healthy glow. Fine lines on his forehead and around his eyes are minimal, and his skin appears firm with good elasticity. His pores are refined, and his under-eye area looks refreshed without noticeable dark circles or puffiness. His overall facial appearance is vibrant, youthful, and well-maintained."
    man_age = f"a photo of person at TARGETAGE years old"

    if person in men:
        target_prompt = man_age
    else:
        target_prompt = woman_age

    for img_filename in image_files:
        img_path = os.path.join(input_folder, img_filename)
        
        # Extract the input age from the image filename
        input_age = extract_input_age(img_path)
        print(f"Processing image: {img_filename}, Extracted input age: {input_age}")

        man_input_prompt = f"A portrait of a {input_age}-year-old {person_ethnicity} man."
        # man_input_prompt = f""
        woman_input_prompt = f"A portrait of a {input_age}-year-old {person_ethnicity} woman."
        # woman_input_prompt = f""
        
        if person in men:
            inversion_prompt = man_input_prompt
        else:
            inversion_prompt = woman_input_prompt

        output_dir = os.path.join(output_root_dir, os.path.splitext(img_filename)[0])
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")
        

        run_edit(img_path, input_age, inversion_prompt, target_prompt, output_dir, person)

    print("Processing complete.")
