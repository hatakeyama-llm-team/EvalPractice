import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_tokenizer_and_model_dir",
                        type=str, required=True)
    parser.add_argument("--huggingface_name", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        input_tokenizer_and_model_dir, device_map="cpu",
              torch_dtype=torch.bfloat16,
        )
    return tokenizer, model


def main() -> None:
    args = parse_arguments()

    # Loads and tests the local tokenizer and the local model.
    local_tokenizer, local_model = load_tokenizer_and_model(
        args.output_tokenizer_and_model_dir)

    local_tokenizer.push_to_hub(args.huggingface_name)
    local_model.push_to_hub(args.huggingface_name)
    return


if __name__ == "__main__":
    main()
