"""
synthetic-lite 3.4.1 — **bug‑fix** JSONL emitter
=================================================
* Fixed unterminated string + stray `(jsonl_str)` call in `qa()`.
* Kept every other feature (Gemini thinking‑budget, YAML config, token guard).

Install / run un­changed:
```bash
python synthetic_lite.py extracted_books/haram_months.txt \
  --config configs/config6.yaml -o test3.jsonl
```
"""
from __future__ import annotations

import json, os, yaml, typer
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

app = typer.Typer(help="synthetic-lite — JSONL QA, Gemini thinking-budget, LangChain v0.3")

###############################################################################
# Helpers
###############################################################################

def load_cfg(path: Path) -> Dict:
    return yaml.safe_load(Path(path).read_text("utf-8"))


def g(cfg: Dict, key: str, default=None):
    return cfg.get("generation", {}).get(key, default)

###############################################################################
# LLM factory
###############################################################################

def make_llm(cfg: Dict, *, thinking_budget: Optional[int] = None):
    prov = cfg["llm"]["provider"]
    pc = cfg[prov]

    if prov in {"api-endpoint", "vllm"}:
        return ChatOpenAI(
            model_name=pc["model"],
            openai_api_base=pc.get("api_base") or os.getenv("OPENAI_API_BASE"),
            openai_api_key=pc.get("api_key") or os.getenv("OPENAI_API_KEY"),
            temperature=g(cfg, "temperature", 0.2),
            max_tokens=g(cfg, "max_tokens", 512),
        )

    if prov == "google-genai":
        think = (thinking_budget if thinking_budget is not None else pc.get("thinking_budget", g(cfg, "thinking_budget")))
        tool_cfg = {"max_thinking_tokens": think} if think else None
        return ChatGoogleGenerativeAI(
            model=pc["model"],
            google_api_key=pc.get("api_key") or os.getenv("GOOGLE_API_KEY"),
            temperature=g(cfg, "temperature", 0.2),
            max_output_tokens=g(cfg, "max_tokens", 512),
            tool_config=tool_cfg,
        )

    typer.echo(f"[error] Unknown provider {prov}", err=True); raise typer.Exit(1)

###############################################################################
# Chunking & guards
###############################################################################

def chunk(text: str, cfg: Dict) -> List[str]:
    return RecursiveCharacterTextSplitter(
        chunk_size=g(cfg, "chunk_size", 4000),
        chunk_overlap=g(cfg, "overlap", 200),
        separators=["\n\n", "\n", " ", ""],
    ).split_text(text)


def est_tok(chars: int) -> int:
    return ceil(chars / 4)


def guard(chunks: List[str], budget: Optional[int]):
    if not budget:
        return
    need = sum(est_tok(len(c)) for c in chunks)
    if need > budget:
        typer.echo(f"[error] Budget {budget} exceeded (need≈{need})", err=True); raise typer.Exit(1)

###############################################################################
# Schema & parser
###############################################################################

class QAPair(BaseModel):
    question: str
    answer: str


class QAPairList(BaseModel):
    pairs: List[QAPair] = Field(description="A list of question and answer pairs.")


# MODIFICATION: Change parser to expect a List of QAPair objects
parser = JsonOutputParser(pydantic_object=QAPairList)

###############################################################################
# QA command
###############################################################################

@app.command()
def qa(
    file: Path = typer.Argument(..., exists=True),
    output: Optional[Path] = typer.Option(None, "-o"),
    config: Path = typer.Option("configs/config.yaml", "--config"),
    token_budget: Optional[int] = typer.Option(None),
    thinking_budget: Optional[int] = typer.Option(None),
):
    if not Path(config).exists():
        # Try to resolve relative to project root (parent of script's dir)
        proj_config = Path(__file__).parent / config
        if proj_config.exists():
            config = proj_config
        else:
            typer.echo(f"[error] Config file not found: {config}", err=True); raise typer.Exit(1)

    cfg = load_cfg(config)
    text = Path(file).read_text("utf-8")
    parts = chunk(text, cfg)
    guard(parts, token_budget or g(cfg, "token_budget"))

    llm = make_llm(cfg, thinking_budget=thinking_budget)
    num_pairs = g(cfg, "num_pairs", 10)

    tmpl = cfg["prompts"]["qa_generation"].replace("{num_pairs}", str(num_pairs)) + "\n\n{format_instructions}"
    prompt = PromptTemplate.from_template(tmpl)

    qa_list = []
    for ck in tqdm(parts, desc="qa"):
        raw = llm.invoke(prompt.format(text=ck, format_instructions=parser.get_format_instructions())).content.strip()
        try:
            # MODIFICATION: Directly parse the output as a list of QAPair objects
            parsed_obj = parser.parse(raw)
            if isinstance(parsed_obj, QAPairList):
                # Convert Pydantic models to dictionaries before appending to qa_list
                qa_list.extend([p.model_dump() for p in parsed_obj.pairs])
            else:
                typer.echo(f"[warning] Parser returned unexpected type or content: {type(parsed_obj)}", err=True)
                typer.echo(f"Raw output: {raw}", err=True)
        except Exception as e:
            # MODIFICATION: Catch any parsing errors and log them
            typer.echo(f"[warning] Failed to parse LLM output for chunk: {e}", err=True)
            typer.echo(f"Raw output: {raw}", err=True)
            continue

    # ---- JSONL emit ----
    if output:
        json_path = output.with_suffix(".json")
        json_path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[info] Wrote {len(qa_list)} pairs to {json_path}")

        # Convert to JSONL from the JSON file
        data = json.loads(json_path.read_text(encoding="utf-8"))
        jsonl_lines = [json.dumps(item, ensure_ascii=False) for item in data]
        jsonl_str = "\n".join(jsonl_lines) + "\n"

        output.write_text(jsonl_str, "utf-8")
        typer.echo(f"[info] Wrote {len(data)} lines → {output}")
    else:
        # For stdout, convert directly from memory
        jsonl_lines = [json.dumps(p, ensure_ascii=False) for p in qa_list]
        jsonl_str = "\n".join(jsonl_lines) + "\n"
        print(jsonl_str)

###############################################################################
# main
###############################################################################

if __name__ == "__main__":
    app()
