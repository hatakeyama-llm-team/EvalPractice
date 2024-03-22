

# fastchatの更新

- 公式版(temp, temperatureのバグ有り, apiがgpt-4でちょっと高価)
~~~
fschat @ git+https://github.com/wandb/FastChat@main
pip install --force-reinstall git+https://github.com/wandb/FastChat@main
~~~

- 改造版
  - temp, temperatureのバグを修正
  - 普段遣い用に､gpt-4-turboを呼べるように変更(OPENAI_MODEL_NAMEを設定)
~~~
pip install --force-reinstall git+https://github.com/hatakeyama-llm-team/FastChat
OPENAI_MODEL_NAME=gpt-4-0125-preview
~~~