# %%
from transformers import AutoTokenizer
from vllm import SamplingParams
import string
from vllm import LLM
from datasets import load_dataset
from tqdm import tqdm
import random
from datetime import datetime
import json
import pandas as pd
import argparse
import os


random.seed(42)

n_batch = 1000

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('jobid', type=int, help='Job ID')
parser.add_argument('model_name', type=str, help='model name')
args = parser.parse_args()

print("Parsed arguments:")
print(args)

job_id = args.jobid
model_name = args.model_name

print(f"Job ID: {job_id}")
print(f"Model Name: {model_name}")

print(job_id,model_name)
#job_id=1

os.environ["CUDA_VISIBLE_DEVICES"] = f"{job_id}"
n_jobs=8
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"data/0520orpo_model/job_{job_id}_{current_time_no_symbols}.jsonl"


# %%
df=pd.read_parquet("data/0520orpo/code_all_10000000000.parquet")
master_records=df.to_dict(orient="records")

# %%
print("init models...")
llm = LLM(model=model_name, trust_remote_code=True)

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

answer_template="\n\n### 応答:\n"


# プロンプトテンプレートの準備
random.seed(42)
random.shuffle(master_records)

#jobで分割
job_size=int(len(master_records)/n_jobs)
records=master_records[job_id*job_size:(job_id+1)*job_size]

cnt = 0
for i in tqdm(range(int(len(records)/n_batch))):
    # プロンプトの準備
    sampled_records = records[cnt*n_batch:(cnt+1)*n_batch]

    prompts = []
    for record in sampled_records:
        q=record["text"]
        q=q[:q.rfind(answer_template)+len(answer_template)]
        prompts.append(tokenizer.encode(q)[:-1])

    # 推論の実行
    outputs = llm.generate(
        # prompts,
        prompt_token_ids=prompts,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=512,
        )
    )
    for i, output in enumerate(outputs):
        sampled_records[i]["model_answer"] = output.outputs[0].text

    with open(out_path, "a") as f:
        for record in sampled_records:
            record.pop("text")
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    cnt += 1


# %%
len(prompts),len(sampled_records),len(outputs)

# %%
outputs

# %%



