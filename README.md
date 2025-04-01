# smolR1

<img src="assets/logo.png" width="400">


reproducing DeepSeek R1 Zero with Qwen2.5-0.5B on two 4090 GPUs


## Setup
Create a virtual enviroment and install dependencies with 
```bash
pip install -r requirements.txt
```


For a two GPU setup, start vLLM first.

```
CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen2.5-0.5B
```

Then run training with

```
accelerate launch --config_file configs/deepspeed/zero3.yaml --num_processes 1 train.py
```

## Acknowledegments

[simpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason)

[Qwen2.5 Math Evaluation](https://github.com/QwenLM/Qwen2.5-Math)
