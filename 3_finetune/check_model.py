from transformers import pipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


#モデル読み込み

#math系を沢山学習させたせたモデル
model_name="/storage5/EvalPractice/model/0602with_halcination_math_-storage5-llm-models-hf-step62160_fin_inst_-storage5-EvalPractice-3_finetune-0524with_halcination_little_codes_synth_eng-inst_parquet_lr_5e-5/checkpoint-14800"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            #quantization_config=bnb_config, 
                                            device_map="auto",
                                            torch_dtype=torch.bfloat16,
                                            )

pipe=pipeline('text-generation',model=model,tokenizer=tokenizer, 
              max_new_tokens=250, 
              repetition_penalty=1.0,
              temperature=0.6,

              )

question_template="以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n"
answer_template="\n\n### 応答:\n"

def gen_prompt(q):
    return f"{question_template}{q}{answer_template}"

questions=[
"たぬきに純粋理性批判は理解できますか?",
"フィボナッチ数列を生成するpythonのcode",
"元気ですか?",
 "日本の首相は?",
"東京科学大学の学長は?",
 "東京科学大学とは?"   ,
"次の四則演算をしなさい\n 1+2+3",
"1+2+3-4はいくつか",
"1+2+3",
"1/2+3を計算せよ",
"計算をせよ: 0.4+1",
"将来的な映画製作者が学ぶべき五つの受賞歴のあるドキュメンタリー映画とそれぞれの背景説明を提案してください。",
]

for question in questions:
    inp=gen_prompt(question)
    print("--------------------")
    print(question)
    res=pipe(inp,max_new_tokens=256)[0]["generated_text"][len(inp):]
    print(res)
    #print(tokenizer.encode(res)[:4])

