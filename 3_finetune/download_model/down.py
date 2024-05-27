from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

#model = AutoModelForCausalLM.from_pretrained('hatakeyama-llm-team/with_halcination_little_codes_ck5200')
#tokenizer = AutoTokenizer.from_pretrained('hatakeyama-llm-team/with_halcination_little_codes_ck5200')

data = load_dataset('hatakeyama-llm-team/data_dpo_trial')