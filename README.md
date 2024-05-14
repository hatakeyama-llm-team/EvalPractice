# setup
~~~
git clone -b gcp https://github.com/hatakeyama-llm-team/EvalPractice.git
~~~

# Finetuning & Evaluationのcode
- 事前学習済みモデルのファインチューニングと評価を一気通貫して行うscript

# Setup
## ファインチューニングのライブラリ群
- llmevalと同じ環境で作ればOKと
~~~
conda create -n llmeval python=3.11 -y
conda activate llmeval
#llm-jp-sftのレポジトリに移動
cd llm-jp-sft
pip install -r requirements.in 
pip install flash-attn --no-build-isolation #flash atten
pip install --upgrade accelerate #accelerateを最新版に

#evalのリポジトリ
cd 4_eval/llm-leaderboard
conda create -n llmeval python=3.11 -y
conda activate llmeval
pip3 install -r llm-leaderboard/requirements.txt

~~~


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
- auto_finetuneなどを実行する
### upload
- 必要に応じて､HuggingFaceにモデルをアップロードします｡
- コマンド例
~~~
python 3_upload.py --output_tokenizer_and_model_dir ../model/llm-jp-llm-jp-13b-v1-0_inst_dolly10000 --huggingface_name llm-jp-llm-jp-13b-v1-0_inst_dolly10000

~~~
