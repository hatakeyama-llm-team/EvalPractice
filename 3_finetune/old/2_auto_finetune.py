import glob
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#print(os.environ["CUDA_VISIBLE_DEVICES"])

inst_path_list = (glob.glob("data/*.jsonl.parquet"))
print(inst_path_list)


lr_list = [
    # "1e-6",
    #"5e-6",
     "1e-5",
     "1e-4",
    # "5e-5",
]

model_name_list=[
"/storage5/llm/models/hf/step62160_fin",
]

for model_name in model_name_list:
    print("train: ",model_name)
    for inst_path in inst_path_list:
        for lr in lr_list:
            out_name = model_name+"_inst_"+inst_path
            out_name = out_name.replace(".jsonl", "").replace(
                "/", "-").replace(".", "-").replace("data-", "")
            out_name = out_name+"_lr_"+lr
            out_path = "../model/"+out_name
            eval_path = inst_path+".eval"

            print(lr)
            print(eval_path)
            print(model_name)
            print(out_path)


        #        --use_peft true
        # --gradient_checkpointing true \

            #マルチgpu
            pre_cmd="accelerate launch --config_file ./llm-jp-sft/configs/accelerate_config_zero1.yaml ./llm-jp-sft/train.py"
            #通常
            #pre_cmd="python ./llm-jp-sft/train.py"

            cmd = f"""{pre_cmd}  \
                --num_train_epochs 3 \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 3 \
                --save_strategy "epoch" \
                --logging_steps 1 \
                --gradient_accumulation_steps 16 \
                --learning_rate {lr} \
                --warmup_ratio 0.1 \
                --lr_scheduler_type cosine \
                --bf16 \
                --data_files {inst_path} \
                --model_name_or_path {model_name} \
                --use_fast False \
                --output_dir {out_path} \
                --instruction_template "\n\n### 指示:\n" \
                --response_template "\n\n### 応答:\n" \
                --use_flash_attention_2 True \
                --gradient_checkpointing true \

            """

            #--load_in_4bit True \
            os.system(cmd)
# --response_template "\n\n### 応答:\n" \
"""
                --peft_target_model mixtral \
                --use_peft True \
                --peft_lora_r 4096 \
                --peft_lora_alpha 4096 \
                --eval_data_files {eval_path} \
"""