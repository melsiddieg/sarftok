import json
from pathlib import Path

INPUT_PATH  = Path("qa_original_fixed.jsonl")          # your source file
OUTPUT_PATH = Path("qa_gemma_chat.jsonl")        # destination

def convert_line(raw_line: str) -> str:
    """
    Convert a single Q-A JSON line into Gemma-style 'messages' JSON.
    Returns the new line as a UTF-8 string (without trailing newline).
    """
    qa = json.loads(raw_line)
    record = {
        "messages": [
            {"role": "user",  "content": qa["question"].strip()},
            {"role": "model", "content": qa["answer"].strip()}
        ]
    }
    return json.dumps(record, ensure_ascii=False)

with INPUT_PATH.open("r", encoding="utf-8") as fin, \
     OUTPUT_PATH.open("w", encoding="utf-8") as fout:
    for raw in fin:
        out_line = convert_line(raw)
        fout.write(out_line + "\n")

print(f"✓ Converted → {OUTPUT_PATH.resolve()}")

