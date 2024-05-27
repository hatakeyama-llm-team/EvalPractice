# upload commandの例

~~~
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path="/storage5/EvalPractice/model/-storage5-llm-models-hf-step62160_fin_inst_all_10000000000-parquet_lr_1e-5/checkpoint-1455"
tokenizer = AutoTokenizer.from_pretrained(path)
#model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu",torch_dtype=torch.bfloat16)
#cpuの場合
model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu",torch_dtype=torch.fp16)
huggingface_name="hatakeyama-llm-team/tanuki_inst_0515test"
tokenizer.push_to_hub(huggingface_name)
model.push_to_hub(huggingface_name)


~~~

#batch ft
cd EvalPractice/3_finetune/

#job1
sbatch --nodelist=slurm0-a3-ghpc-[12] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run.py data/0524clean_halcination_little_codes 1_0524clean_halcination_little_codes

#job2
sbatch --nodelist=slurm0-a3-ghpc-[13] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run.py data/0524with_halcination_little_codes 2_0524with_halcination_little_codes

#job1,2再開
sbatch --nodelist=slurm0-a3-ghpc-[12] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run_resume.py data/0524clean_halcination_little_codes 1_0524clean_halcination_little_codes
sbatch --nodelist=slurm0-a3-ghpc-[12] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run_resume.py data/0524with_halcination_little_codes 2_0524with_halcination_little_codes


#job3
sbatch --nodelist=slurm0-a3-ghpc-[12] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run.py data/0524with_halcination_little_codes_synth_eng 3_0524with_halcination_little_codes_synth_eng

#job4 mathを盛りだくさん
sbatch --nodelist=slurm0-a3-ghpc-[13] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run.py data/0524with_halcination_little_codes_synth_eng_math 4_0524with_halcination_little_codes_synth_eng_math

#job4 再開
sbatch --nodelist=slurm0-a3-ghpc-[14] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run_resume.py data/0524with_halcination_little_codes_synth_eng_math 4_0524with_halcination_little_codes_synth_eng_math


#job5 clean & multiturnも300文字以上
sbatch --nodelist=slurm0-a3-ghpc-[12] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run.py data/0524_5_dataset_clean_halcination_longer_multiturn 0524_5_dataset_clean_halcination_longer_multiturns

#job6 merged model
sbatch --nodelist=slurm0-a3-ghpc-[13] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run_merge.py data/0524with_halcination_little_codes_synth_eng_math 6_0524merged_with_halcination_synth_eng_math


#merged, 日英synth
sbatch --nodelist=slurm0-a3-ghpc-[13] --gpus-per-node=8 --time=30-00:00:00 -c 200 run.sh 0524ft_run_merge.py 0524with_halcination_little_codes_synth_eng 0524with_halcination_little_codes_synth_eng
