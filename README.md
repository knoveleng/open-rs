# Open RS

This repository hosts the code and datasets for the **Open RS** project, accompanying the paper *Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn’t*. The project explores enhancing reasoning capabilities in small large language models (LLMs) using reinforcement learning (RL) under resource-constrained conditions.

We focus on a 1.5-billion-parameter model, `DeepSeek-R1-Distill-Qwen-1.5B`, trained on 4 NVIDIA A40 GPUs (48 GB VRAM each) within 24 hours. By adapting the Group Relative Policy Optimization (GRPO) algorithm and leveraging a curated, compact mathematical reasoning dataset, we conducted three experiments to assess performance and behavior. Key findings include:

- Significant reasoning improvements, e.g., AMC23 accuracy rising from 63% to 80% and AIME24 reaching 46.7%, outperforming `o1-preview`.
- Efficient training with just 7,000 samples at a cost of $42, compared to thousands of dollars for baseline models.
- Challenges like optimization instability and length constraints with extended training.

These results showcase RL-based fine-tuning as a cost-effective approach for small LLMs, making reasoning capabilities accessible in resource-limited settings. We open-source our code, models, and datasets to support further research.

![Performance Metrics](assets/performances.png)

## Resources

### Models
- [Open-RS1](https://huggingface.co/knoveleng/Open-RS1)
- [Open-RS2](https://huggingface.co/knoveleng/Open-RS2)
- [Open-RS3](https://huggingface.co/knoveleng/Open-RS3)
- Additional models in training: [knoveleng/OpenRS-GRPO](https://huggingface.co/knoveleng/OpenRS-GRPO), [quyanh/OpenRS-GRPO](https://huggingface.co/quyanh/OpenRS-GRPO)

### Datasets
- [open-s1](https://huggingface.co/datasets/knoveleng/open-s1)
- [open-deepscaler](https://huggingface.co/datasets/knoveleng/open-deepscaler)
- [open-rs](https://huggingface.co/datasets/knoveleng/open-rs) (used in Experiments 2 and 3)

### Collection
- [Open-RS Collection](https://huggingface.co/collections/knoveleng/open-rs-67d940abc201a7e7f252ca4e)

## Installation

### Prerequisites
Install `uv` for managing virtual environments:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Set up a virtual environment with Python 3.11:
```bash
uv venv openr1 --python 3.11
source openr1/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy
```

### Dependencies
Install `vLLM` and `FlashAttention`:
```bash
uv pip install vllm==0.7.2
uv pip install setuptools
uv pip install flash-attn --no-build-isolation
```

> **Note**: This installs PyTorch `v2.5.1`, which is required for `vLLM` compatibility. Using a different version may cause issues.

Install additional dependencies based on your use case:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

### Authentication
Log in to Hugging Face and Weights & Biases:
```bash
huggingface-cli login
wandb login
```

### Git LFS
Ensure Git LFS is installed for model/dataset management:
```bash
git-lfs --version
```
If not installed:
```bash
sudo apt-get install git-lfs
```

## Training

Train models using a YAML config with 4 GPUs (set `num_processes=3`):
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  src/open_r1/grpo.py \
  --config recipes/grpo.yaml
```

For Experiment 3, add the `cosine_max_len` parameter:
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  src/open_r1/grpo.py \
  --config recipes/grpo.yaml \
  --cosine_max_len 3584
```

## Evaluation

Evaluate models using `lighteval` with custom tasks in `src/open_r1/evaluate.py`. For single-GPU setups:
```bash
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# Example: AIME 2024
TASK=aime24
lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"
```

> **Important**: Set `max_model_length=32768` to match `max_new_tokens`, or `lighteval` will fail.

For multi-GPU evaluation with data parallelism:
```bash
NUM_GPUS=4
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"
```

Alternatively, use the evaluation script:
```bash
sh eval.sh
```
Modify tasks in `eval.sh` (line 8) as needed.

### Performance Highlights
- **Open-RS1**: 53.0% avg. score
- **Open-RS2**: 55.7% avg. score, 80.0% on AMC23
- **Open-RS3**: 56.3% avg. score, 46.7% on AIME24 (outperforms `o1-preview` at 44.6%)
- Competitive MATH-500 scores; Minerva lags behind 7B models.

![Performance Metrics](assets/performances.png)

### Cost Efficiency
Our approach uses 7,000 samples (42,000 total outputs) and costs ~$42 on 4x A40 GPUs in 24 hours, compared to:
- 7B models: `Qwen2.5-7B-SimpleRL` ($1,633), `Eurus-2-7B-PRIME` ($1,088)
- 1.5B models: `DeepScaleR-1.5B-Preview` ($3,629), `Still-3-1.5B-Preview` ($2,268)

![7B Model Costs](assets/costs-7b.png)  
![1.5B Model Costs](assets/costs-1.5b.png)

## Acknowledgements
Thanks to the Hugging Face team for their [open-r1](https://github.com/huggingface/open-r1) project.

## Citation
Coming soon.
