from transformers import AutoModelForCausalLM, AutoTokenizer

for path in [
"hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160",
"hatakeyama-llm-team/Tanuki_pretrained_stage5_step58800",
"hatakeyama-llm-team/Tanuki_pretrained_stage4_step49700",
"hatakeyama-llm-team/Tanuki_pretrained_stage3_step43400",
"hatakeyama-llm-team/Tanuki_pretrained_stage2_step37800",
"hatakeyama-llm-team/Tanuki_pretrained_stage1_step30800",

]:
    try:
        print(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="cpu")
    except Exception as e:
        print(e)
        continue