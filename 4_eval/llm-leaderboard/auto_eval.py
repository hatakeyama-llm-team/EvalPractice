# %%
import yaml
import glob
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# %%
# YAMLファイルを読み込む
with open('configs/config_eval.yaml', 'r') as file:
    data = yaml.safe_load(file)
# model_path_list = glob.glob("../../data/model/*")
model_path_list = [
    "llm-jp/llm-jp-13b-v1.0",
     "kanhatakeyama/llm-jp-llm-jp-13b-v1-0_inst_dolly10000",
     "kanhatakeyama/llm-jp-llm-jp-13b-v1-0_inst_ichikara_1500",
     "kanhatakeyama/llm-jp-llm-jp-13b-v1-0_inst_oasst40000",
]

for model_path in model_path_list:
    model_name = model_path.split("/")[-1]
    run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+model_name
    data["wandb"]["run_name"] = run_name
    data["model"]["pretrained_model_name_or_path"] = model_path
    data["tokenizer"]["pretrained_model_name_or_path"] = model_path
    data["metainfo"]["basemodel_name"] = model_name

    # YAMLファイルに書き込む
    with open('configs/config_eval_.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)

    # 評価を自動実行
    cmd = "python scripts/run_eval_modif_batch.py"
    os.system(cmd)

# %%
