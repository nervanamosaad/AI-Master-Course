# Milestone 2 – Automated Regression Evaluation Suite

## Overview
This project implements an automated evaluation suite for a locally running language model using Ollama.

The evaluation domain is Water Resources in Egypt.

## Main Features
- 34 Golden Test Samples.
- Programmatic assertion-based validation.
- LLM-as-a-Judge evaluation using a 1–5 scoring rubric.
- Judge calibration using predefined examples.
- Automated PASS/FAIL results.
- Overall pass rate and category-based summary.
- Detailed evaluation results saved in JSON format.

## Requirements
- Python 3
- Ollama
- `gemma2:2b`
- Python package: `requests`

## Run the Project

1. Start Ollama.
2. Make sure the `gemma2:2b` model is available.
3. Install the required package:

   `pip install requests`

4. Run:

   `python evaluation.py`

## Evaluation Configuration
- Answer Model: `gemma2:2b`
- Judge Model: `gemma2:2b`
- Golden Samples: 34
- Minimum Judge Score: 4/5
- Required Pass Rate: 80%

## Output
The evaluation results are saved in `results.json`.