# Finetuning & Evaluationのcode
- 事前学習済みモデルのファインチューニングと評価を一気通貫して行うscript

# Setup
## 標準ライブラリ群
~~~
cd 4_eval/llm-leaderboard
conda create -n llmeval python=3.11 -y
conda activate llmeval

pip3 install -r llm-leaderboard/requirements.txt
pip install langchain-anthropic
~~~

## fastchatの更新
- 公式版は24/3/23時点で
  - temp, temperatureのtypoバグ有り
  - また､apiがgpt-4でちょっと高価

- 公式版
~~~
fschat @ git+https://github.com/wandb/FastChat@main
pip install --force-reinstall git+https://github.com/wandb/FastChat@main
~~~

- 改造版
  - temp, temperatureのバグを修正
  - 普段遣い用に､gpt-4-turboを呼べるように変更(OPENAI_MODEL_NAMEを設定)
~~~
pip install --force-reinstall git+https://github.com/hatakeyama-llm-team/FastChat
OPENAI_MODEL_NAME=gpt-4-0125-preview
~~~

## wandb関連
- [config](./4_eval/llm-leaderboard/configs/config_eval.yaml)を修正する
  - 例
~~~
entity: "kanhatakeyamas" 
project: "llmeval" 
run_name: "test1" 
~~~


## 実行
### ファインチューニング
- 適当な仮想環境を作っておくこと｡
- デフォルトはフルパラのファインチューニングなので､VRAMに注意
~~~
cd 3_finetune
dataset_file="./llm-jp-sft/data/example.jsonl" #instruction dataset
input_model="stockmark/gpt-neox-japanese-1.4b" #base model
input_model="cyberagent/open-calm-7b"
output_dir="../data/model/test"

# Finetunes the pretrained model.
python ./llm-jp-sft/train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --data_files ${dataset_file} \
    --model_name_or_path ${input_model} \
    --output_dir ${output_dir} \
    --instruction_template "### 指示:" \
    --response_template "### 応答:" \
    #--use_peft true # loraを使う場合
    #2>&1 | tee ${log_path}/${host}_${current_time}.log
~~~

### 評価
~~~
conda activate llmeval
cd 4_eval/llm-leaderboard
python scripts/run_eval_modif.py
~~~


