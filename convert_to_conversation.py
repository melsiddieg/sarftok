import json
import argparse
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file of Q-A pairs to Gemma-style chat format."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input JSONL file (e.g., qa_original_fixed.jsonl)"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to the output JSONL file (e.g., qa_gemma_chat.jsonl)"
    )

    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path.resolve()}")
        return

    try:
        with input_path.open("r", encoding="utf-8") as fin, \
             output_path.open("w", encoding="utf-8") as fout:
            for i, raw in enumerate(fin):
                try:
                    out_line = convert_line(raw)
                    fout.write(out_line + "\n")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {i+1} in {input_path.name}: {raw.strip()}")
                except KeyError as e:
                    print(f"Warning: Skipping line {i+1} due to missing key '{e}' in {input_path.name}: {raw.strip()}")
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return

    print(f"✓ Converted {i+1 if 'i' in locals() else 0} lines → {output_path.resolve()}")

if __name__ == '__main__':
    main()
