# Import libraries
import csv
import json         # read Ollama response
import subprocess   # for nvidia-smi
import time

import requests    


# Settings; model, context size, tokens number
MODEL = "gemma2:2b"
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192]
N_PREDICT = 50              # max. no. of tokens generated each time
TIMEOUT = 900               # max. timeout

OUTPUT_FILE = "results.csv"
URL = "http://localhost:11434/api/generate"        # Ollama Generate API

COLUMNS = [
    "context_length",
    "status",
    "prompt_tokens",
    "generated_tokens",
    "ttft_s",
    "avg_inter_chunk_ms",
    "throughput_tok_s",
    "vram_before_mb",
    "vram_after_mb",
    "vram_increase_mb",
    "error"
]


def get_vram_mb():
    out = subprocess.check_output(
        [
            "nvidia-smi",           # it provides GPU monitoring information (memory usage, directly from the NVIDIA driver)
            "--query-gpu = memory.used",
            "--format = csv, noheader, nounits"
        ],
        text=True           # return as text not Bytes
    )

    # current GPU memory usage in MB
    return int(out.strip().splitlines()[0])     # spaces, lines, first line


def unload_model():             # Unload the model from memory before starting a new test
    response = requests.post(
        URL,
        json={"model": MODEL, "keep_alive": 0},
        timeout=60
    )
    response.raise_for_status()     # error handling
    time.sleep(5)


def build_prompt(context_length):
    sentence = "The quick brown fox jumps over the lazy dog. "

    target_tokens = context_length - N_PREDICT - 100        # Context Window contains both i/p & o/p
    repeats = max(1, target_tokens // 10)               # suppose that repeated sentence = 10 Tokens

    return sentence * repeats + "\nSummarize the text above."


def run_test(context_length):          # [1024, 2048, 4096, 8192]

    unload_model()
    vram_before = get_vram_mb()

    # Warm-up: load the model before measuring TTFT.
    warmup = requests.post(
        URL,
        json={
            "model": MODEL,
            "prompt": "Hi",
            "stream": False,
            "options": {
                "num_ctx": context_length,
                "num_predict": 1
            }
        },
        timeout=TIMEOUT
    )
    warmup.raise_for_status()

    # Prepare the actual test request.
    payload = {
        "model": MODEL,
        "prompt": build_prompt(context_length),
        "stream": True,
        "options": {
            "num_ctx": context_length,
            "num_predict": N_PREDICT,
            "temperature": 0
        }
    }

    chunk_times = []
    final = {}

    start = time.perf_counter()

    with requests.post(
        URL,
        json=payload,
        stream=True,
        timeout=TIMEOUT
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line)            # each JSON line to Python Dictionary

            if "error" in chunk:
                raise RuntimeError(chunk["error"])

            if chunk.get("response"):
                chunk_times.append(time.perf_counter())

            if chunk.get("done"):
                final = chunk           # Ollama statistics; Tokens, times

    if not final or not chunk_times:
        raise RuntimeError("No generated response received from Ollama.")

    vram_after = get_vram_mb()

    # Calculate TTFT and average inter-chunk latency.
    ttft = chunk_times[0] - start

    gaps = [
        b - a
        for a, b in zip(chunk_times, chunk_times[1:])
    ]

    avg_latency = (
        sum(gaps) / len(gaps) * 1000
        if gaps else 0
    )

    # Get generation statistics from Ollama.
    generated_tokens = final.get("eval_count", 0)
    eval_duration = final.get("eval_duration", 0)

    throughput = (
        generated_tokens / (eval_duration / 1e9)
        if eval_duration > 0 else 0
    )

    return {
        "context_length": context_length,
        "prompt_tokens": final.get("prompt_eval_count", 0),
        "generated_tokens": generated_tokens,
        "ttft_s": round(ttft, 3),
        "avg_inter_chunk_ms": round(avg_latency, 2),
        "throughput_tok_s": round(throughput, 2),
        "vram_before_mb": vram_before,
        "vram_after_mb": vram_after,
        "vram_increase_mb": vram_after - vram_before
    }


def main():
    # run all context length tests and save the results

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()

        for context_length in CONTEXT_LENGTHS:

            print(f"\nTesting context length: {context_length}")

            try:
                row = run_test(context_length)
                row["status"] = "ok"

                print(
                    f"TTFT: {row['ttft_s']} s | "
                    f"Latency: {row['avg_inter_chunk_ms']} ms | "
                    f"Throughput: {row['throughput_tok_s']} tok/s | "
                    f"VRAM increase: {row['vram_increase_mb']} MB"
                )

            except Exception as error:

                row = {
                    "context_length": context_length,
                    "status": "failed",
                    "error": str(error)
                }

                print(f"FAILED: {error}")

            # Save each result immediately.
            writer.writerow({
                column: row.get(column, "")
                for column in COLUMNS
            })

            file.flush()            # write now

    try:
        unload_model()
    except Exception:
        pass

    print(f"\nDone. Results saved in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
