import glob
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#print(os.environ["CUDA_VISIBLE_DEVICES"])

inst_path_list = (glob.glob("data/*.jsonl"))
print(inst_path_list)
model_name = "llm-jp/llm-jp-13b-v1.0"
# model_name = "tokyotech-llm/Swallow-MS-7b-v0.1"

lr_list = [
    # "1e-6",
    "5e-6",
    # "1e-5",
    # "5e-5",
]
for inst_path in inst_path_list:
    for lr in lr_list:
        out_name = model_name+"_inst_"+inst_path
        out_name = out_name.replace(".jsonl", "").replace(
            "/", "-").replace(".", "-").replace("data-", "")
        out_name = out_name+"_lr_"+lr
        out_path = "../model/"+out_name
        eval_path = inst_path+".eval"

        cmd = f"""python ./llm-jp-sft/train.py \
            --num_train_epochs 3 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 3 \
            --save_strategy "epoch" \
            --logging_steps 100 \
            --gradient_accumulation_steps 2 \
            --learning_rate {lr} \
            --warmup_ratio 0.1 \
            --lr_scheduler_type cosine \
            --bf16 \
            --data_files {inst_path} \
            --eval_data_files {eval_path} \
            --model_name_or_path {model_name} \
            --output_dir {out_path} \
            --instruction_template "### 指示:\n" \
            --response_template "### 応答:\n" \
            --gradient_checkpointing true \
        """
    #        --use_peft true
    # --gradient_checkpointing true \
        cmd = f"""torchrun --nproc_per_node=1 llm-jp-sft/train_multi.py \
            --num_train_epochs 3 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 3 \
            --save_strategy "epoch" \
            --logging_steps 100 \
            --gradient_accumulation_steps 2 \
            --learning_rate {lr} \
            --warmup_ratio 0.1 \
            --lr_scheduler_type cosine \
            --bf16 \
            --data_files {inst_path} \
            --eval_data_files {eval_path} \
            --model_name_or_path {model_name} \
            --output_dir {out_path} \
            --instruction_template "### 指示:\n" \
            --response_template "### 応答:\n" \
            --gradient_checkpointing true \
        """

        cmd = f"""deepspeed --num_gpus 8 ./llm-jp-sft/train.py \
            --num_train_epochs 3 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 3 \
            --save_strategy "epoch" \
            --logging_steps 100 \
            --gradient_accumulation_steps 2 \
            --learning_rate {lr} \
            --warmup_ratio 0.1 \
            --lr_scheduler_type cosine \
            --bf16 \
            --data_files {inst_path} \
            --eval_data_files {eval_path} \
            --model_name_or_path {model_name} \
            --output_dir {out_path} \
            --instruction_template "### 指示:\n" \
            --response_template "### 応答:\n" \
            --gradient_checkpointing true \
        """
        cmd = f"""accelerate launch --num_processes 8 ./llm-jp-sft/train.py  \
            --num_train_epochs 3 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 3 \
            --save_strategy "epoch" \
            --logging_steps 100 \
            --gradient_accumulation_steps 2 \
            --learning_rate {lr} \
            --warmup_ratio 0.1 \
            --lr_scheduler_type cosine \
            --bf16 \
            --data_files {inst_path} \
            --eval_data_files {eval_path} \
            --model_name_or_path {model_name} \
            --output_dir {out_path} \
            --instruction_template "### 指示:\n" \
            --response_template "### 応答:\n" \
            --gradient_checkpointing true \
        """
        os.system(cmd)


    #使わない
