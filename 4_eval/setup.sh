cd 3_finetune
# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
git clone https://github.com/hotsuyuki/llm-jp-sft
cd llm-jp-sft
git fetch origin
git checkout refs/tags/ucllm_nedo_dev_v20240208.1.0

cd ../

#eval
#evalように新しく環境を作る
cd 4_eval
git clone https://github.com/matsuolab/llm-leaderboard.git