import glob
import os

inst_path_list = glob.glob("data/*.jsonl")
model_name = "llm-jp/llm-jp-13b-v1.0"


for inst_path in inst_path_list:
    out_name = model_name+"_inst_"+inst_path
    out_name = out_name.replace(".jsonl", "").replace(
        "/", "-").replace(".", "-").replace("data-", "")
    out_path = "..//model/"+out_name

    cmd = f"""python ./llm-jp-sft/train.py \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --learning_rate 1e-5 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --bf16 \
        --data_files {inst_path} \
        --model_name_or_path {model_name} \
        --output_dir {out_path} \
        --instruction_template "### 指示:" \
        --response_template "### 応答:" \
        --use_peft true
    """

    os.system(cmd)
