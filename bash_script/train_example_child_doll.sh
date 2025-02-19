# Config

# The pre-trained model used in CoRe
model_name="stabilityai/stable-diffusion-2-1-base"
# The directory to save the fine-tuned model
output_dir="./models/"
# The directory containing real images of the target concept
instance_dir="./examples/child_doll"
# The name of the concept
base_name="child_doll"
# The category of the target concept
category_token="doll"
seed=2147483647
# Important:
# Select the appropriate prompts file based on the type of training object
prompts_file_path='./regularization_prompt_flie/Animate.txt'

# The directory to save the output of the first stage
ti_output_dir="${output_dir}/learned_embeds/${base_name}"
# The directory to save the output of the second stage
unet_output_dir="${output_dir}/learned_models/${base_name}"

# Train
# Stage 1
accelerate launch train.py \
    --train_ti \
    --pretrained_model_name_or_path=$model_name \
    --object_token="${base_name}" \
    --initialize_token=${category_token} \
    --save_step 50 \
    --instance_data_dir=$instance_dir \
    --output_dir=$ti_output_dir \
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
    --seed $seed\
    --prompts_file_path="${prompts_file_path}" \
    --embedding_cosine_weight "1.5e-4" \
    --attention_mean_weight 0.05 \
    --rescale_start_step 120 \
    --rescale_stop_step 180 


# Stage 2
accelerate launch train.py \
    --pretrained_model_name_or_path=$model_name  \
    --resume_from_checkpoint "latest" \
    --save_step 200 \
    --only_save_checkpoint \
    --checkpointing_steps 200 \
    --train_unet \
    --embedding_path="${ti_output_dir}/learned_embeds_steps_300.bin" \
    --instance_data_dir=$instance_dir \
    --output_dir=$unet_output_dir \
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
    --seed $seed \
    --checkpoints_total_limit 1 \

# inference
python inference.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --checkpoint_path="${unet_output_dir}/checkpoint-1000" \
    --learned_embedding_path="${ti_output_dir}/learned_embeds_steps_300.bin" \
    --prompt="A {} on the beach" \
    --save_dir="${output_dir}/output" \
    --infer_scheduler="DPMSolverMultistepScheduler" \
    --num_images_per_prompt 32 \
    --infer_batch_size 32 \
    --seed $seed