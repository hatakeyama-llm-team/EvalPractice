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

## データセットのダウンロード
- [こちらを実行](./3_finetune/1_prepare_inst_dataset.py)

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

## jglue_evalのfix bug (3/23時点)
- [jglue_eval](4_eval/llm-leaderboard/scripts/jglue_eval.py)の305,321行目付近に､cfg=cfgを入れる必要有り
- このレポジトリでは対応済み

## wandb関連
- 評価用に､[新規プロジェクトをwandb上で作ります](https://wandb.ai/new-project)
- [config](./4_eval/llm-leaderboard/configs/config_eval.yaml)を修正します｡
  - entity,projectは､上記のurlから作成したもの
  - run_nameは適当
~~~
entity: "kanhatakeyamas" 
project: "llmeval" 
run_name: "test1" 
~~~


## 実行
### ファインチューニング
- 適当な仮想環境を作っておくこと｡
- use_peft trueとすると､LoRAで学習が進む
  - llama, llm-jp以外のモデルの場合､adapterを自分で指定する必要あり
    - [train.py](3_finetune/llm-jp-sft/train.py)の50行目付近を編集する
  - use_peft falseの場合は､通常のフルパラファインチューニング
    - 1bモデルで70GBほどのVRAMを使います
    - 7bの場合､A100(80GB) x2でもVRAMは足りないので注意

~~~
cd 3_finetune
dataset_file="./llm-jp-sft/data/example.jsonl" #instruction dataset
input_model="stockmark/gpt-neox-japanese-1.4b" #base model
input_model="llm-jp/llm-jp-13b-v1.0"
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
    --instruction_template "### 質問：" \
    --response_template "### 回答：" \
    --use_peft true # loraを使う場合
    #2>&1 | tee ${log_path}/${host}_${current_time}.log
~~~

### upload
- HuggingFaceにモデルをアップロードします｡
- コマンド例
~~~
python 3_upload.py --output_tokenizer_and_model_dir ../model/llm-jp-llm-jp-13b-v1-0_inst_dolly10000 --huggingface_name llm-jp-llm-jp-13b-v1-0_inst_dolly10000

~~~

### 評価
- GPT4-turboを使った場合､1回の評価に2-3ドル程度?
~~~

OPENAI_MODEL_NAME=gpt-4-0125-preview # gpt-4 turboを使う場合(デフォルトはgpt-4)

conda activate llmeval
cd 4_eval/llm-leaderboard
python scripts/run_eval_modif.py
~~~



# 自動のモデル構築と評価(仮)
- データセットのサイズなどを変えながら､自動評価していきます
- [ファインチューニング](./3_finetune/2_auto_finetune.py)
- [評価](./4_eval/llm-leaderboard/auto_eval.py)


# TODO
- instruct集
- 数学系
  - https://huggingface.co/datasets/kunishou/OpenMathInstruct-1-1.8m-ja
- プログラミング
  - https://huggingface.co/datasets/kunishou/amenokaku-code-instruct?
- 一般
  - https://huggingface.co/datasets/kunishou/HelpSteer-35k-ja?row=16
  - https://huggingface.co/datasets/sudy-super/CoTangent
  - https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset

- 酒
  - https://huggingface.co/datasets/yuiseki/sake_qa