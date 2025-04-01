# Qwen2.5 Math Evaluation

adapted from https://github.com/hkust-nlp/simpleRL-reason
who adapted from https://github.com/QwenLM/Qwen2.5-Math

```bash
# install in seperate environment

# simpleRL instructions
cd examples/simplelr_math_eval
pip uninstall latex2sympy2 -y
cd latex2sympy
pip install -e . --use-pep517
pip install Pebble
pip install sympy==1.12
pip install antlr4-python3-runtime==4.11.1
pip install timeout-decorator
pip install jieba
cd ..

# official Qwen2.5 Math instructions
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3


bash sh/eval_simple.sh

```

simpleRL-Zoo eval code seems broken.

I measured with official Qwen Math repo and will port it over soon.
