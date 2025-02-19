# Config

# The pre-trained model used in CoRe
model_name="stabilityai/stable-diffusion-2-1-base"
# The directory to save the fine-tuned model
output_dir="./models/"
# The directory containing real images of the target concept
instance_dir=""
# The name of the concept
base_name=""
# The category of the target concept
category_token=""
seed=2147483647

# The prompt you want to optimize for
prompt="A {} on the beach"
# The directory to save the output
ti_output_dir="${output_dir}/test_time_optimization/${base_name}/${prompt// /_}"

# The checkpoint directory obtained from the previous train
checkpoint_dir=""
# The embedding path obtained from the previous train
embedding_path=""

accelerate launch train.py \
    --train_ti \
    --pretrained_model_name_or_path=$model_name \
    --object_token="${base_name}" \
    --initialize_token=${category_token} \
    --save_step 5 \
    --instance_data_dir=$instance_dir \
    --output_dir=$ti_output_dir \
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
    --seed $seed\
    --prompt="${prompt}" \
    --load_from_checkpoint $checkpoint_dir \
    --embedding_path $embedding_path \
    --embedding_cosine_weight "1.5e-4" \
    --attention_mean_weight 0.05 \
    --ignore_diffusion_loss 
