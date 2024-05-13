~~~
git clone git@github.com:matsuolab/llm-leaderboard.git
conda create -n llmeval2
pip3 install -r llm-leaderboard/requirements.txt

export LANG=ja_JP.UTF-8
# https://wandb.ai/settings#api
export WANDB_API_KEY=<your WANDB_API_KEY>
# 運営から共有されたkeyを指定してください
export OPENAI_API_KEY=<your OPENAI_API_KEY>
# if needed, please login in huggingface(private設定にしている際など)
huggingface-cli login
~~~