"""
ARC Challenge — Ollama Version
================================
Runs the ARC Challenge dataset through Ollama locally.

Pipeline:
  1. Download & load the ARC Challenge dataset
  2. Build MCQ prompts
  3. Run inference via Ollama
  4. Score responses
  5. Save results to JSONL

Requirements:
  pip install ollama tqdm datasets

Ollama must be running:
  ollama serve          (in a separate terminal, or already running in the tray)

Pull the model first (one-time):
  ollama pull qwen2.5:1.5b
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import ollama
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID    = "llama3.2:1b"               # Change to any installed Ollama model
DATA_PATH   = "data/arc_challenge_test.jsonl"
OUTPUT_PATH = "results/arc_results.jsonl"
MAX_TOKENS  = 512
SAVE_EVAL   = True                          # Set False for private test set


# ── Dataset Loader ─────────────────────────────────────────────────────────────
def download_arc_dataset(save_path: str):
    """Download ARC Challenge test set from HuggingFace and save as JSONL."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Run: pip install datasets")
        sys.exit(1)

    print("Downloading ARC Challenge dataset from HuggingFace...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(save_path)
    print(f"Saved to {save_path}\n")


def load_arc_dataset(path: str) -> list:
    """
    Load ARC Challenge JSONL and convert to the format this script expects.

    ARC format:
      {
        "id": "Mercury_7175875",
        "question": "Which factor ...",
        "choices": { "text": ["A", "B", "C", "D"], "label": ["A","B","C","D"] },
        "answerKey": "B"
      }

    Converted to:
      {
        "id": "Mercury_7175875",
        "question": "Which factor ...",
        "options": ["choice A text", "choice B text", ...],
        "answer": "B"
      }
    """
    data = []
    for line in open(path):
        item = json.loads(line)
        choices = item["choices"]

        # Re-map answer key: ARC sometimes uses "1","2","3","4" instead of "A","B","C","D"
        label_to_letter = {
            label: chr(65 + i)
            for i, label in enumerate(choices["label"])
        }
        answer_letter = label_to_letter.get(item["answerKey"], item["answerKey"])

        data.append({
            "id":       item["id"],
            "question": item["question"],
            "options":  choices["text"],     # flat list of answer strings
            "answer":   answer_letter,       # single letter: A, B, C, or D
        })
    return data


# ── System Prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_MCQ = (
    "You are an expert at answering science and reasoning questions. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


# ── Prompt Builder ─────────────────────────────────────────────────────────────
def build_prompt(question: str, options: list) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a MCQ question."""
    labels    = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
    return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(data: list) -> list:
    """Run Ollama inference on all questions."""
    responses = []

    print(f"Running inference on {len(data)} questions with model '{MODEL_ID}'...")
    print("(Ollama runs one question at a time)\n")

    for item in tqdm(data, desc="Generating"):
        system, user = build_prompt(item["question"], item["options"])

        try:
            response = ollama.chat(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options={
                    "num_predict":    MAX_TOKENS,
                    "temperature":    0.6,
                    "top_p":          0.95,
                    "top_k":          20,
                    "repeat_penalty": 1.0,
                },
            )
            text = response["message"]["content"].strip()
        except Exception as e:
            print(f"\nError on question id={item.get('id')}: {e}")
            text = ""

        responses.append(text)

    return responses


# ── Scoring ────────────────────────────────────────────────────────────────────
def extract_letter(text: str) -> str:
    """Extract answer letter from \\boxed{X}, or fall back to last uppercase letter."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_results(data: list, responses: list) -> list:
    results = []
    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        predicted = extract_letter(response)
        correct   = predicted == item["answer"].strip().upper()

        results.append({
            "id":        item["id"],
            "gold":      item["answer"],
            "predicted": predicted,
            "response":  response,
            "correct":   correct,
        })
    return results


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(results: list):
    total   = len(results)
    correct = sum(r["correct"] for r in results)
    acc     = correct / total * 100 if total else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS — ARC Challenge")
    print("=" * 50)
    print(f"  Correct : {correct:4d} / {total:4d}  ({acc:.2f}%)")
    print("=" * 50)


# ── Save Results ───────────────────────────────────────────────────────────────
def save_results(results: list, output_path: str, save_eval: bool = True):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            if save_eval:
                record = {
                    "id":        r["id"],
                    "gold":      r["gold"],
                    "predicted": r["predicted"],
                    "response":  r["response"],
                    "correct":   r["correct"],
                }
            else:
                record = {
                    "id":        r["id"],
                    "predicted": r["predicted"],
                    "response":  r["response"],
                }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} records to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Download dataset if not already present
    if not Path(DATA_PATH).exists():
        download_arc_dataset(DATA_PATH)

    # 2. Load dataset
    print(f"Loading dataset from {DATA_PATH}...")
    data = load_arc_dataset(DATA_PATH)
    print(f"Loaded {len(data)} questions\n")

    # Preview sample
    sample = data[0]
    print("── Sample question ──")
    print(f"  Q : {sample['question']}")
    for i, opt in enumerate(sample["options"]):
        print(f"  {chr(65+i)} : {opt}")
    print(f"  Answer: {sample['answer']}\n")

    # 3. Run inference
    # To test on a small subset first, uncomment the line below:
    # data = data[:20]
    responses = run_inference(data)

    # 4. Score
    results = score_results(data, responses)

    # 5. Summary
    print_summary(results)

    # 6. Save
    save_results(results, OUTPUT_PATH, save_eval=SAVE_EVAL)


if __name__ == "__main__":
    main()
