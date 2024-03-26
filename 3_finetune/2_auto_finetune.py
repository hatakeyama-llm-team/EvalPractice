import glob
import os

inst_path_list = glob.glob("data/*.jsonl")
print(inst_path_list)
model_name = "llm-jp/llm-jp-13b-v1.0"
# model_name = "tokyotech-llm/Swallow-MS-7b-v0.1"


for inst_path in inst_path_list:
    out_name = model_name+"_inst_"+inst_path
    out_name = out_name.replace(".jsonl", "").replace(
        "/", "-").replace(".", "-").replace("data-", "")
    out_path = "../model/"+out_name
    eval_path=inst_path+".eval"

    cmd = f"""python ./llm-jp-sft/train.py \
        --num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --learning_rate 2e-5 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --bf16 \
        --data_files {inst_path} \
        --eval_data_files {eval_path} \
        --model_name_or_path {model_name} \
        --output_dir {out_path} \
        --instruction_template "### 指示:" \
        --response_template "### 応答:" \
        --gradient_checkpointing true \
    """

#        --use_peft true
# --gradient_checkpointing true \
    os.system(cmd)
