# CoRe: Context-Regularized Text Embedding Learning for Text-to-Image Personalization
> Recent advances in text-to-image personalization have enabled high-quality and controllable image synthesis for user-provided concepts. However, existing methods still struggle to balance identity preservation with text alignment. Our approach is based on the fact that generating prompt-aligned images requires a precise semantic understanding of the prompt, which involves accurately processing the interactions between the new concept and its surrounding context tokens within the CLIP text encoder. To address this, we aim to embed the new concept properly into the input embedding space of the text encoder, allowing for seamless integration with existing tokens. We introduce Context Regularization (CoRe), which enhances the learning of the new concept's text embedding by regularizing its context tokens in the prompt. This is based on the insight that appropriate output vectors of the text encoder for the context tokens can only be achieved if the new concept's text embedding is correctly learned. CoRe can be applied to arbitrary prompts without requiring the generation of corresponding images, thus improving the generalization of the learned text embedding.
> Additionally, CoRe can serve as a test-time optimization technique to further enhance the generations for specific prompts. Comprehensive experiments demonstrate that our method outperforms several baseline methods in both identity preservation and text alignment.
<img src='assets/teaser.png'>

## Update
-  **2024.11.28**: The repository released!
-  **2025.12.10**: The paper was accepted by AAAI 2025!
-  **2025.2.19**: The code is officially released.

## Environment setup

To set up the environment, please run:
```
conda env create -f environment.yaml
pip install accelerate
accelerate config
```


## Usage
It is recommended to refer to the scripts in the `./bash_script` folder for usage.

### Train
You can simply run the `./bash_script/train_example_child_doll.sh` script to train an example concept, "child_doll".

```
# Stage 1
accelerate launch train.py \
    --train_ti \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \ 
    --object_token="${base_name}" \ # The name of the concept
    --initialize_token=${category_token} \ # The category of the target concept
    --save_step 50 \
    --instance_data_dir=$instance_dir \ # The directory containing real images of the target concept
    --output_dir=$ti_output_dir \ # The directory to save the trained embedding output of the first stage
    --instance_prompt="a photo of a {}" \
    --resolution=512 \
    --train_batch_size 6 \
    --gradient_accumulation_steps=1 \
    --learning_rate "5e-3" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --validation_prompt="a photo of a {}" \
    --validation_steps 50 \
    --max_train_steps 300 \
    --seed 2147483647 \
    --prompts_file_path="${prompts_file_path}" \ # Select the appropriate prompts file based on the type of training object
    --prompts_cosine_weight "1.5e-4" \
    --ti_attn_mean_weight 0.05 \
    --rescale_start_step 120 \
    --rescale_stop_step 180 


# Stage 2
accelerate launch train.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"  \
    --resume_from_checkpoint "latest" \
    --save_step 200 \
    --only_save_checkpoint \
    --checkpointing_steps 200 \
    --train_unet \
    --embedding_path="${ti_output_dir}/learned_embeds_steps_300.bin" \ # The trained embedding path of the first stage
    --instance_data_dir=$instance_dir \ # The directory containing real images of the target concept
    --output_dir=$unet_output_dir \ # The directory to save the output of the second stage
    --instance_prompt="a photo of a {}" \
    --resolution=512 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate="2e-6" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --validation_prompt="a photo of a {}" \
    --validation_steps 200 \
    --max_train_steps 1000 \
    --seed 2147483647 \
    --checkpoints_total_limit 1 \
```
**Note:**
- Select the appropriate prompts file based on the type of training object for `--prompts_file_path`. For example, use `'./regularization_prompt_flie/Animate.txt'` for Animate concepts and `'./regularization_prompt_flie/Inanimate.txt'` for Inanimate concepts.

### Inference
To run inference on a learned embedding, you can run:
```
python inference.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --checkpoint_path="$checkpoint_path" \ # The checkpoint directory obtained from the previous train
    --learned_embedding_path=$learned_embedding_path \ # The embedding path obtained from the previous train
    --prompt="A {} in Times Square" \ # The prompt you want to inference, {} is a placeholder for the concept
    --save_dir=$output_dir \ # The directory to save the pictures
    --infer_scheduler="DPMSolverMultistepScheduler" \
    --seed 2147483647
```
**Note:**
- For convenience, you can either specify a path to a text file with '--prompt_file', where each line contains a prompt. For example:
```
A {} in a snowy mountain landscape
A {} with the Eiffel Tower in the background
A {} with a wheat field in the background
```
### Test-time optimization
<img src='assets/test-time.png'>
You can also use Test-time optimization to fine-tune a specific prompt.

```
python train.py \
    --train_ti \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --object_token="${base_name}" \ # The name of the concept
    --initialize_token=${category_token} \ # The category of the target concept
    --save_step 5 \
    --instance_data_dir=$instance_dir \ # The directory containing real images of the target concept
    --output_dir=$ti_output_dir \ # The directory to save the output
    --instance_prompt="a photo of a {}" \
    --resolution=512 \
    --train_batch_size 6 \
    --gradient_accumulation_steps=1 \
    --learning_rate "5e-4" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --validation_prompt="a photo of a {}" \
    --validation_steps 5 \
    --max_train_steps 10 \
    --seed 2147483647 \
    --prompt="A {} as a cowboy is dancing on the stage" \ # The prompt you want to optimize for
    --load_from_checkpoint $checkpoint_dir \ # The checkpoint directory obtained from the previous train
    --embedding_path $embedding_path \ # The embedding path obtained from the previous train
    --embedding_cosine_weight "1.5e-4" \
    --attention_mean_weight 0.05 \
    --ignore_diffusion_loss 
```

## Acknowledgements

This project builds upon the work of several other repositories. We would like to express our gratitude to the following projects for their contributions:
- [AttnDreamBooth (NeurIPS 2024)](https://github.com/lyuPang/AttnDreamBooth): Official Implementation of "AttnDreamBooth: Towards Text-Aligned Personalized Text-to-Image" by Lianyu Pang, Jian Yin, Baoquan Zhao, Qing Li and Xudong Mao.
- [Diffusers-Textual Inversion](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion): A implementation of An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion
- [Dreambooth-Stable-Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion): Implementation of Dreambooth (https://arxiv.org/abs/2208.12242) with Stable Diffusion
- [Diffusers](https://github.com/huggingface/diffusers): A library for state-of-the-art pretrained diffusion models.

---

## References

```
@article{wu2024core,
  title={CoRe: Context-Regularized Text Embedding Learning for Text-to-Image Personalization},
  author={Wu, Feize and Pang, Yun and Zhang, Junyi and Pang, Lianyu and Yin, Jian and Zhao, Baoquan and Li, Qing and Mao, Xudong},
  journal={arXiv preprint arXiv:2408.15914},
  year={2024}
}
···
