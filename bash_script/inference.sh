# This is the path to the embeddings learned in the first stage.
learned_embedding_path=""
# This is the path to the fine-tuned models trained in the third stage.
checkpoint_path=""
# The directory where the generated images will be saved.
output_dir=""
seed=2147483647


# inference
python inference.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --checkpoint_path=$checkpoint_path \
    --learned_embedding_path=$learned_embedding_path \
    --prompt="A {} on the beach" \
    --save_dir=$output_dir \
    --infer_scheduler="DPMSolverMultistepScheduler" \
    --num_images_per_prompt 32 \
    --infer_batch_size 32 \
    --seed $seed