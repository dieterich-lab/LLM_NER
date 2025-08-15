import json
import json_repair

import sys
import re
from typing import Dict, Literal

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- customize your allowed labels here ----
Label = Literal["PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME"]

# Pydantic v1/v2 compatibility (RootModel in v2, __root__ in v1)
try:
    from pydantic import RootModel  # v2
    class Entities(RootModel[Dict[str, Label]]):
        pass
    def validate_payload(payload: dict) -> Dict[str, str]:
        return Entities(payload).root
except Exception:
    from pydantic import BaseModel  # v1
    class Entities(BaseModel):
        __root__: Dict[str, Label]
    def validate_payload(payload: dict) -> Dict[str, str]:
        return Entities(__root__=payload).__root__

MODEL_ID = "openai/gpt-oss-20b"  

SYSTEM_PROMPT = """You are given a text, and need to find occurrences of the following entities in it:
PERSON: person names (first and/or family name)
LOCATION: physical locations (countries, cities, addresses, etc.)
ORGANIZATION: organizations (companies, associations, schools, etc.)
DATE_TIME: dates or times.

Return ONLY a single JSON object that maps each exact surface mention to one of these labels.
No comments, no prose, no code fences.

Example:
Input text: 'I am Ole Nordmann and I am born on October 7, 1965 in Mo i Rana'
Response: {"Ole Nordmann":"PERSON","October 7, 1965":"DATE_TIME","Mo i Rana":"LOCATION"}"""

USER_TEXT = (
    "Hello there, what's up? I'm Ole Nordmann. I used to live in Oslo, "
    "but I have now moved to Mo i Rana. This message was written on September 22, "
    "and I work for Telenor."
)

def build_input(tokenizer, text: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Input text: '{text}'"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    # Fallback to a plain prompt if no chat template is available
    prompt = SYSTEM_PROMPT + "\n\n" + f"Input text: '{text}'\nResponse:"
    return tokenizer(prompt, return_tensors="pt").input_ids

def extract_json_str(text: str) -> str:
    """
    Try to extract the first top-level JSON object. Keeps things simple.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in the model output.")
    candidate = text[start : end + 1].strip()
    return candidate

def parse_json_or_repair(json_str: str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired = json_repair.repair_json(json_str)
        return json.loads(repaired)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None, device_map="auto"
    )

    input_ids = build_input(tokenizer, USER_TEXT).to(model.device)

    # Decoding knobs: low temperature to reduce chatter; cap tokens sensibly
    gen_kwargs = dict(
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    # Slice off the prompt part
    gen_ids = out[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    try:
        json_str = extract_json_str(raw)
        payload = parse_json_or_repair(json_str)
    except Exception as e:
        print("Failed to parse JSON from model output.\n--- RAW OUTPUT ---\n", raw, file=sys.stderr)
        raise

    # Validate with Pydantic (structure + allowed labels)
    try:
        validated = validate_payload(payload)  # Dict[str, Label]
    except Exception as e:
        print("JSON failed Pydantic validation. Error:", e, file=sys.stderr)
        print("Payload was:", json.dumps(payload, indent=2), file=sys.stderr)
        raise

    # Success → pretty-print and save
    print(json.dumps(validated, indent=2, ensure_ascii=False))
    with open("entities.json", "w", encoding="utf-8") as f:
        json.dump(validated, f, ensure_ascii=False, indent=2)
    print("\nSaved to entities.json")

if __name__ == "__main__":
    main()
