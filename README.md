# Greta Plugins

Current plugins:
vLLM: 0.30.0
Huggingface Transformers: 5.18.0

## Install

### From source

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.'
```

### From remote

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install git+https://github.com/Garbo-AI/greta-plugin
```

## Usage

### vLLM Serve

```bash
VLLM_PLUGINS=greta vllm serve garboab/Greta-1b-Preview \
  --model-impl vllm \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --mamba-cache-mode none \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --reasoning-parser greta \
  --reasoning-parser-plugin /path/to/greta_vllm/greta_vllm/reasoning_parser.py \
  --tool-call-parser greta \
  --tool-parser-plugin /path/to/greta_vllm/greta_vllm/tool_parser.py \
  --chat-template /path/to/greta_vllm/chat_template.jinja \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8
```


### vLLM Python

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/greta-checkpoint",
    model_impl="vllm",
    dtype="bfloat16",
    tensor_parallel_size=1,
    enforce_eager=True,
    enable_prefix_caching=False,
    mamba_cache_mode="none",
    enable_chunked_prefill=True,
    max_num_batched_tokens=1024,
    max_num_seqs=8,
)
outputs = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=64))
print(outputs[0].outputs[0].text)
```