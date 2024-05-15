from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
path="/storage5/EvalPractice/model/-storage5-llm-models-hf-step62160_fin_inst_all_10000000000-parquet_lr_1e-5/checkpoint-1455"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu",torch_dtype=torch.float16)
huggingface_name="hatakeyama-llm-team/tanuki_inst_0515test"
tokenizer.push_to_hub(huggingface_name)
while True:
    try:
        model.push_to_hub(huggingface_name)
        break
    except Exception as e:
        print(e)
        time.sleep(10)

