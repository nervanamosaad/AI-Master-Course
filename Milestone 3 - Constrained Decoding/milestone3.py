
import json
import time
import ollama
import outlines

from pydantic import BaseModel, ConfigDict, ValidationError
from typing import Literal


MODEL = "gemma2:2b"


# Record Ollama responses to collect generation statistics
# to calculate throughput
class RecordingClient(ollama.Client):

    last_response = None

    def chat(self, *args, **kwargs):
        self.last_response = super().chat(*args, **kwargs)
        return self.last_response


client = RecordingClient()

# Connect Outlines to the local Ollama model
outlines_model = outlines.from_ollama(client, MODEL)


# Define Output Schema
class WaterReport(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="forbid"
    )

    water_level_risk: Literal["Low", "Medium", "High"]
    rainfall_risk: Literal["Low", "Medium", "High"]
    temperature_risk: Literal["Low", "Medium", "High"]
    summary: str


# Prepare Test Samples to test and compare 2 methods

samples = [
    "A region has low water levels, little rainfall, and high temperatures.",
    "A region has normal water levels, moderate rainfall, and moderate temperatures.",
    "A region has high water levels, heavy rainfall, and low temperatures.",
    "A region has low water levels, moderate rainfall, and high temperatures.",
    "A region has high water levels, heavy rainfall, and high temperatures."
]


# Prepare Prompt: each sample of 5, to prompt

def prepare_prompt(sample):

    return f"""
Analyze the following water resources report:

{sample}

Classify the risks using Low, Medium, or High.

Return one JSON object with exactly these four fields:
- water_level_risk
- rainfall_risk
- temperature_risk
- summary

Example format:
{{
    "water_level_risk": "Low",
    "rainfall_risk": "Low",
    "temperature_risk": "High",
    "summary": "A short explanation."
}}

IMPORTANT:
- Return valid JSON only.
- Do not include explanations outside the JSON.
- Do not use Markdown or code fences.
- Do not add any extra fields.
"""


# Method 1: Unconstrained Generation with Retries
def unconstrained_generation(prompt, max_retries=3):

    total_tokens = 0
    total_duration = 0

    syntax_errors = 0
    schema_errors = 0

    # Keep conversation history for corrective retries
    messages = [
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):

        # Send request to Gemma
        response = client.chat(
            model=MODEL,
            messages=messages
        )

        # Collect statistics from every attempt
        total_tokens += response.eval_count or 0
        total_duration += response.eval_duration or 0

        raw_output = response.message.content

        try:
            data = json.loads(raw_output)

        except json.JSONDecodeError:
            syntax_errors += 1

            if syntax_errors == 1:
                print("Raw model output:", repr(raw_output))

            print("Invalid JSON syntax. Retrying...")

            # Add previous response and explain the error
            # Corrective Retry / Feedback-based Retry
            messages.append({
                "role": "assistant",
                "content": raw_output
            })

            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was rejected because it "
                    "was not valid raw JSON. Return the same information "
                    "as a JSON object only. Do not use Markdown, "
                    "```json, or code fences. Start with { and end with }."
                )
            })

            continue

        try:
            # Check result using Pydantic
            result = WaterReport.model_validate(data)

            return (
                result,
                attempt + 1,
                syntax_errors,
                schema_errors,
                total_tokens,
                total_duration
            )

        except ValidationError as error:
            schema_errors += 1

            print("Invalid schema. Retrying...")

            # Add previous response and explain the schema error
            messages.append({
                "role": "assistant",
                "content": raw_output
            })

            messages.append({
                "role": "user",
                "content": (
                    "Your previous JSON did not match the required schema. "
                    "Correct these validation errors:\n"
                    f"{error}\n"
                    "Return a corrected JSON object only. "
                    "Do not use Markdown or code fences."
                )
            })

    return (
        None,
        max_retries,
        syntax_errors,
        schema_errors,
        total_tokens,
        total_duration
    )


# Method 2: Constrained Generation using Outlines
def constrained_generation(prompt):

    # Pass Prompt & Pydantic Model to Outlines
    raw_output = outlines_model(prompt, WaterReport)

    # Validate the generated JSON against the schema
    result = WaterReport.model_validate_json(raw_output)

    return result


# Run Benchmark for all samples with the 2 methods

def run_benchmark():

    baseline_times = []
    constrained_times = []

    baseline_tokens = 0
    baseline_duration = 0

    constrained_tokens = 0
    constrained_duration = 0

    baseline_success = 0
    constrained_success = 0

    total_retries = 0
    total_syntax_errors = 0
    total_schema_errors = 0

    print("\nStarting Milestone 3 Benchmark\n")

    for i, sample in enumerate(samples, start=1):

        prompt = prepare_prompt(sample)

        # Unconstrained Generation
        start = time.perf_counter()

        (
            baseline_result,
            attempts,
            syntax_errors,
            schema_errors,
            tokens,
            duration
        ) = unconstrained_generation(prompt)

        baseline_times.append(time.perf_counter() - start)

        total_retries += attempts - 1
        total_syntax_errors += syntax_errors
        total_schema_errors += schema_errors

        baseline_tokens += tokens
        baseline_duration += duration

        if baseline_result is not None:
            baseline_success += 1

        # Constrained Generation
        start = time.perf_counter()

        # Clear the previous response statistics
        client.last_response = None

        try:
            constrained_result = constrained_generation(prompt)
            constrained_success += 1

        except Exception as error:
            constrained_result = None
            print("Constrained generation error:", error)

        constrained_times.append(time.perf_counter() - start)

        # Collect statistics from the Outlines request
        response = client.last_response

        if response is not None:
            constrained_tokens += response.eval_count or 0
            constrained_duration += response.eval_duration or 0

        print(f"Sample {i} completed.")

    # Calculate Metrics

    avg_baseline_latency = sum(baseline_times) / len(samples)
    avg_constrained_latency = sum(constrained_times) / len(samples)

    baseline_throughput = (
        baseline_tokens / (baseline_duration / 1_000_000_000)
        if baseline_duration > 0 else 0
    )

    constrained_throughput = (
        constrained_tokens / (constrained_duration / 1_000_000_000)
        if constrained_duration > 0 else 0
    )

    # Print Results

    print("\nBenchmark Results")

    print("\nUnconstrained Generation:")
    print("Successful Samples:", baseline_success, "/", len(samples))
    print("Average Latency:", round(avg_baseline_latency, 3), "seconds")
    print("Throughput:", round(baseline_throughput, 2), "tokens/sec")
    print("Total Retries:", total_retries)
    print("JSON Syntax Errors:", total_syntax_errors)
    print("Schema Errors:", total_schema_errors)

    print("\nConstrained Generation:")
    print("Successful Samples:", constrained_success, "/", len(samples))
    print("Average Latency:", round(avg_constrained_latency, 3), "seconds")
    print("Throughput:", round(constrained_throughput, 2), "tokens/sec")


# Execute

if __name__ == "__main__":
    run_benchmark()
