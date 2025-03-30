# smolR1

<img src="assets/logo.png" width="400">



Create a virtual enviroment and install with `pip install -r requirements.txt`.


For a two GPU setup, start vLLM first.

```
CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen2.5-0.5B
```

Then run training with

```
accelerate launch --config_file configs/deepspeed/zero3.yaml --num_processes 1 train_peft.py
```