# setup
~~~
git clone -b gcp https://github.com/hatakeyama-llm-team/EvalPractice.git
~~~

# Finetuning & Evaluationのcode
- 事前学習済みモデルのファインチューニングと評価を一気通貫して行うscript

# Setup
## ファインチューニングのライブラリ群
- 適当な仮想環境を作る｡
- 最近のpytorch､transformers､SFT､wandbライブラリなどを入れればOK｡(雑ですみません)
  - 以下の, llmevalと同じ環境でOK

## 評価の標準ライブラリ群
~~~
cd 4_eval/llm-leaderboard
conda create -n llmeval python=3.11 -y
conda activate llmeval

pip3 install -r llm-leaderboard/requirements.txt
#pip install langchain-anthropic
~~~

## データセットのダウンロード
- [こちらを実行](./3_finetune/1_prepare_inst_dataset.py)


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
### upload
- 必要に応じて､HuggingFaceにモデルをアップロードします｡
- コマンド例
~~~
python 3_upload.py --output_tokenizer_and_model_dir ../model/llm-jp-llm-jp-13b-v1-0_inst_dolly10000 --huggingface_name llm-jp-llm-jp-13b-v1-0_inst_dolly10000

~~~

### 評価
- MTBenchにはGPT4での評価が必要
  - GPT4-turboを使った場合､1回の評価に2-3ドル程度?
- MTBenchをやらない場合は､[run_eval](./4_eval/llm-leaderboard/scripts/run_eval.py)の65行目付近にある､mtbenchをコメントアウトする
- OPENAI_MODEL_NAMEを環境変数に設定すると､評価モデルを選べる(公式は､turboでないGPT-4)
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

# ftの構築メモ 0507
~~~
conda create -n llmeval2 python=3.11 -y
#llm-jp-sftのレポジトリに移動
cd llm-jp-sft
pip install -r requirements.in 
pip install mergoo #MoE用
pip install flash-attn --no-build-isolation #flash atten
pip install --upgrade accelerate #accelerateを最新版に
~~~