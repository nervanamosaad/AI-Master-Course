# Milestone 1 – Local Model Inference & Hardware Profiler

## Overview
This milestone implements a local inference profiling tool using Ollama and the Gemma 2 2B model.

The profiler evaluates model performance at different context lengths: 1024, 2048, 4096, and 8192 tokens.

## Measurements
- Time to First Token (TTFT)
- Average Inter-chunk Latency
- Generation Throughput (tokens/second)
- GPU VRAM usage before and after inference

## Technologies
- Python
- Ollama
- Gemma 2 2B
- NVIDIA SMI

## How to Run

1. Install Python and Ollama.
2. Download the model:

```bash
ollama pull gemma2:2b
```

3. Install the required Python package:

```bash
pip install requests
```

4. Run the profiler:

```bash
python ollama_profiler.py
```

The results are saved in `results.csv`.

## Notes
The latency measurement represents average inter-chunk arrival time from Ollama streaming responses, not exact per-token latency.