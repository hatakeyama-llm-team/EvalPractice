# %%
import os
from transformers import TrainingArguments
from trl import DPOTrainer

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


run_name = "with_halcination_little_codes_ck5200__dpo-lr5e-6-beta0.01"
# Load model and tokenizer
model_id = "hatakeyama-llm-team/with_halcination_little_codes_ck5200"
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%


train_dataset = load_dataset("hatakeyama-llm-team/dpo_data_for_tanuki", split="train")
eval_dataset = load_dataset("hatakeyama-llm-team/dpo_data_for_tanuki", split="test")

# %%



os.environ["WANDB_PROJECT"] = "huggingface"
os.environ["WANDB_NAME"] = run_name

training_args = TrainingArguments(
    output_dir= f"./{run_name}",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size= 2,
    gradient_accumulation_steps=64,
    learning_rate=5.0e-7,
    warmup_ratio=0.1,
    num_train_epochs=5,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    bf16=True,
    dataloader_num_workers=24,
    weight_decay = 0.0,
    lr_scheduler_type="cosine",
    gradient_checkpointing=False,
    run_name=run_name,
    report_to="wandb",
    optim="adamw_torch",
)

dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    beta=0.01,
    loss_type="sigmoid",
    max_prompt_length=925,
    max_length=1150,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# %%

dpo_trainer.train()
dpo_trainer.push_to_hub(f'hatakeyama-llm-team/{run_name}')


