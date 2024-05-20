# %%
from transformers import pipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name="kanhatakeyama/0516tanuki_lr5e5_epoch1"
#model_name="../X_merge/merged_models/0517"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


# %%
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            #quantization_config=bnb_config, 
                                            device_map="auto",
                                            torch_dtype=torch.bfloat16,
                                            )


# %%
#model.load_adapter(model_path)

# %%
pipe=pipeline('text-generation',model=model,tokenizer=tokenizer, 
              max_new_tokens=512, 
              repetition_penalty=1.2,
              temperature=0.6,
              #repetition_penalty=1.,
              #temperature=0.1,
              #top_p=1.0,
              #top_k=0.
              )


# %%
import pandas as pd
question_template="以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n"
answer_template="\n\n### 応答:\n"
answer_template="\n\n### 応答:\n"

def gen_prompt(q):
    return f"{question_template}{q}{answer_template}"


def dump(res_list):
    df=pd.DataFrame(res_list)
    df.to_csv("data/raw_ans.csv",index=False)

#dump(res_list)

# %%
questions=[
"計算してください 1+1はいくつですか｡",
"1+2+3-4=",
"ドラえもんの友達はだれですか?",
"レイリー散乱とはなんですか",
"hello!",
"今の天気は晴れで25℃、明日の天気は雨23℃です｡この結果をjsonで出力してください｡ ",
"iphoneの評価は4で感想は高すぎ, androidの評価は5でgoogle最高!, というレビューをyamlで出力してください｡",
"たぬきに純粋理性批判は理解できますか?",
 "日本の首相は?",
"東京科学大学の学長は?",
]


# %%
res_list=[]


for question in questions:
    inp=gen_prompt(question)
    print("--------------------")
    print(question)
    res=pipe(inp,)[0]["generated_text"][len(inp):]
    print(res)

    d={
        "q":question,
        "model_a":res,
        "ref_a":"",
        "database":"original",
    }
    res_list.append(d)


dump(res_list)

# %%
import pandas as pd
from tqdm import tqdm

chat_path="bench_data/0514llmchat.csv"
df=pd.read_csv(chat_path)
df=df[df["question"].notnull()]
records=df.to_dict(orient="records")

for record in tqdm(records):
    question=record["question"]
    inp=gen_prompt(question)
    print("--------------------")
    print(question)
    res=pipe(inp,)[0]["generated_text"][len(inp):]
    print(res)

    d={
        "q":question,
        "model_a":res,
        "ref_a":record["answer"],
        "database":"llmchat",
    }

    res_list.append(d)


dump(res_list)

# %%
from datasets import load_dataset
#minnade
m_ds=load_dataset("minnade/chat-daily",split="train")

id_to_content={}
for record in m_ds:
    id_to_content[record["id"]]=record["body"]

done_questions=[]
for record in tqdm(m_ds):
    if record["role"]=="assistant":
        q=id_to_content[record["parent_id"]]
        a=record["body"]
        if a is None:
            continue
        if len(a)<4:
            continue
        #questions.append((q,a))
        if q in done_questions:
            continue
        done_questions.append(q)
        inp=gen_prompt(q)
        print("--------------------")
        print(q)
        res=pipe(inp,)[0]["generated_text"][len(inp):]
        print(res)


        d={
                "q":q,
                "model_a":res,
                "ref_a":a,
                "database":"minnnade",
            }

    res_list.append(d)


dump(res_list)

# %%
# elyza

m_ds=load_dataset("elyza/ELYZA-tasks-100",split="test")


for record in tqdm(m_ds):
    q=record["input"]
    #questions.append((q,a))
    inp=gen_prompt(q)
    print("--------------------")
    print(question)
    res=pipe(inp,)[0]["generated_text"][len(inp):]
    print(res)


    d={
            "q":q,
            "model_a":res,
            "ref_a":record["output"],
            "database":"elyza",
        }

    res_list.append(d)

dump(res_list)

# %%



