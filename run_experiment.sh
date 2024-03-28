Llava_Checkpoint="/Projects/marie/weird_skin_spot_challenge/LLaVA-7b"

# 1. Convert original data to format that fits Llava
#    In our case, the language should be Chinese
python scipts/convert_train_data_to_llava.py \
    --input_data_path ./data/train.json \
    --output_data_path ./data/llava_train.json \
    --language "zh"

# 2. Run Training
#    Make sure you set batch size & gradient accumulation steps according
#    to your GPU setup: train_batch_size x gradient_accumulation_steps x num_gpus = 16
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    ./LLaVA-Med/llava/train/train_mem.py \
    --model_name_or_path $Llava_Checkpoint \
    --data_path ./data/llava_train.json \
    --image_folder ./data/images_train \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./LLAVA-ZH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --save_steps 5000 \

# 3. Convert the test set so it fits LLaVA's format
python ./scipts/convert_test_data_to_llava.py \
    --input_data_path ./data/test.json \
    --output_data_path ./data/llava_test.json \
    --language "zh"

# 4. Use the newly created checkpoint to run QA inference
python ./LLaVA-Med/llava/eval/model_vqa.py \
    --model-name ./LLAVA-ZH \
    --question-file ./data/llava_test.json \
    --image-folder ./data/images_test \
    --answers-file ./predictions/llava_zh.jsonl

# 5. Convert LLaVA's predictions back to the challenge target format
python ./scipts/convert_preds_for_challenge.py \
    --input_data_path ./predictions/llava_zh.jsonl \
    --output_data_path ./predictions/prediction.json \
    --language "zh"