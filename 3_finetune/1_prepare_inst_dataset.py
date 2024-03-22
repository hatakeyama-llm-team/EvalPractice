# %%
# instruciton datasetを今回のSFT形式に変換する

# %%
from datasets import load_dataset
import json

# %% [markdown]
# # ichikara

# %%
# 理研のichikara dataset (CC-NC-NDライセンス)
dataset = load_dataset("p1atdev/ichikara-instruction", '20231221-003')["train"]

# %%
print(len(dataset))

# %%
"""
本番はこっち｡
以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい
"""

question_template = "### 指示：以下の質問に答えなさい。 ### 質問："
answer_template = " ### 回答："

# %%
n_instructions = 1500
ichikara_list = []

for n_instructions in [10, 100, 500, 1000, 1500]:
    output_path = f"data/ichikara_{n_instructions}.jsonl"
    with open(output_path, "w") as f:
        f.write("")
    loader = iter(dataset)
    for i in range(n_instructions):
        original_record = next(loader)
        q = original_record["text"]
        a = original_record["output"]
        text = f"{question_template}{q}{answer_template}{a}"
        with open(output_path, "a") as f:
            line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(line+"\n")

        ichikara_list.append(line)

ichikara_list = list(set(ichikara_list))

# %% [markdown]
# # dolly

# %%
d_dataset = load_dataset("kunishou/databricks-dolly-15k-ja")["train"]

# %%
d_dataset[0]

# %%
question_template2 = "### 指示：以下の質問に答えなさい。"
dolly_list = []
for n_instructions in [500, 1000, 2000, 5000, 10000]:
    output_path = f"data/dolly{n_instructions}.jsonl"
    with open(output_path, "w") as f:
        f.write("")
    loader = iter(d_dataset)
    for i in range(n_instructions):
        original_record = next(loader)
        if "input" in original_record:
            inp = original_record["input"]
        else:
            inp = ""
        q = original_record["instruction"]
        a = original_record["output"]
        if inp == "":
            text = f"{question_template}{q}{answer_template}{a}"
        else:
            text = f"{question_template2}{inp} ### 質問：{q}{answer_template}{a}"
        with open(output_path, "a") as f:
            line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(line+"\n")
            dolly_list.append(line)

dolly_list = list(set(dolly_list))

# %%
# 両方
n_ichi = 1500
n_dolly = 8000
output_path = f"data/dolly_{n_dolly}_ichi{n_ichi}.jsonl"
with open(output_path, "w") as f:
    f.write("")
for i in range(n_dolly):
    with open(output_path, "a") as f:
        line = json.dumps({"text": dolly_list[i]}, ensure_ascii=False)
        f.write(line+"\n")
for i in range(n_ichi):
    with open(output_path, "a") as f:
        line = json.dumps({"text": ichikara_list[i]}, ensure_ascii=False)
        f.write(line+"\n")


# %%
