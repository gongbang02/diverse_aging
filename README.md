<div align="center">
  
# Personalized Multi-Condition Age Transformation with Diffusion Transformer

[Bang Gong](https://scholar.google.com/citations?hl=zh-CN&user=PeXg3OYAAAAJ&view_op=list_works&authuser=1&gmla=ANZ5fUPsSh7Re8_0I5yXEqZTVVY4wzCDs5Knfxt9-1oJcFQU8XosSgADRnRaO1ooNCnIGMhPKP4bHOPRUlVBibdn8TFREUxabQda18tzcTo), [Luchao Qi](https://luchaoqi.com/), [Roni Sengupta](https://www.cs.unc.edu/~ronisen/)

University of North Carolina at Chapel Hill  

</div>

<p>
We propose a novel method to solve an underexplored problem, which is to perform image-based age transformation under different conditions specified by text. Our Flux-based approach can effectively preserves the input subject’s identity while accounting for multiple aging possibilities, without requiring any per-subject training on personalized datasets. 
</p>



# 🖼️ Code for Image Editing

For image editing, we employ FLUX as the backbone, which comprises several double blocks and single blocks. Double blocks independently modulate text and image features, while single blocks concatenate these features for unified modulation. In this architecture, our method shares features within both the double blocks and the single blocks in order to enhance identity preservation of the edited image.

To perform image editing with your own image, run
```bash
python edit.py \
    --source_prompt 'a photo of male/female at {INPUT AGE} years old' \
    --target_prompt YOUR_TARGET_PROMPT \
    --guidance 2 \
    --source_img_dir /path-to-your-image \
    --num_steps 15 \
    --inject 3 \
    --name 'flux-dev' \
    --offload \
    --output_dir /path-to-output-dir
```

We have provided examples for image editing using FLUX as the backbone, which can be found <a href="./scripts">Here</a>.</strong>


# 🎨 Gallery


## Image Stylization
Here's a comparison of our method with other Flux-based image editing methods, as well as age transformation methods
<p align="center">
<img src="/repo_figures/visual_results.png" width="1080px"/>
</p>




# Acknowledgements
We thank [FLUX](https://github.com/black-forest-labs/flux/tree/main) and [RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit/tree/main) for their clean codebase.

# Contact
The code in this repository is still being reorganized. Errors that may arise during the organizing process could lead to code malfunctions or discrepancies from the original research results. If you have any questions or concerns, please send emails to gongbang@cs.unc.edu.
