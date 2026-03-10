# %% [markdown]
# ## Cell 1: Install Dependencies

# %%
# !pip install -q cerebras-cloud-sdk "datasets>=2.18.0" "pandas>=2.0.0" "tqdm>=4.66.0"

# %% [markdown]
# ## Cell 2: Imports

# %%
import sys
import os
import json
import re
import time
import gc
from typing import Dict, List, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from cerebras.cloud.sdk import Cerebras
from cerebras.cloud.sdk import RateLimitError as CerebrasRateLimitError

sys.stdout.reconfigure(encoding='utf-8')

# %% [markdown]
# ## Cell 3: Configuration & Model Loading

# %%
class Config:
    # Model Settings
    MODEL_NAME = "gpt-oss-120b"
    CEREBRAS_API_KEY = "csk-f4hwyffn5n99f6j83wn6njteh66m6rwkrxrn5mvp4wrh622e"

    # Generation settings
    MAX_NEW_TOKENS = 16384
    TEMPERATURE = 1.0
    TOP_P = 1.0

    # Benchmark settings
    NUM_PUZZLES = 5
    SEED = 42
    CONFIDENCE_THRESHOLD = 0.9

    # Rate-limit handling
    CALL_DELAY_S = 2.0
    MAX_RETRIES  = 9999

config = Config()

client = Cerebras(api_key=config.CEREBRAS_API_KEY)
print(f'Cerebras client ready — model: {config.MODEL_NAME}')

# %% [markdown]
# ## Cell 4: LLM Engine with Token Tracking

# %%
_total_input_tokens  = 0
_total_output_tokens = 0
_total_elapsed_ms    = 0.0


class LLMEngine:
    def __init__(self, openai_client, config):
        self.client = openai_client
        self.config = config
        self.call_log = []
        print("[LLMEngine] Initialized with token+latency tracking")

    def generate(self, prompt: str, max_tokens: int = None,
                 temperature: float = None, system_prompt: str = None) -> Dict:
        global _total_input_tokens, _total_output_tokens, _total_elapsed_ms
        if max_tokens is None:
            max_tokens = self.config.MAX_NEW_TOKENS
        if temperature is None:
            temperature = self.config.TEMPERATURE

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.config.TOP_P,
                    reasoning_effort="low",
                )
                break
            except CerebrasRateLimitError as e:
                wait = 2 ** attempt
                print(f"[LLMEngine] 429 RateLimitError (attempt {attempt+1}/{self.config.MAX_RETRIES}). "
                      f"Waiting {wait}s before retry...")
                time.sleep(wait)
                if attempt == self.config.MAX_RETRIES - 1:
                    raise
        latency_ms = (time.perf_counter() - start) * 1000

        raw_text = response.choices[0].message.content or ""
        raw_text = raw_text.strip()

        text = raw_text
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        usage = response.usage
        prompt_tokens     = usage.prompt_tokens     if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        _total_input_tokens  += prompt_tokens
        _total_output_tokens += completion_tokens
        _total_elapsed_ms    += latency_ms

        # ===== DEBUG: Show prompt and response =====
        call_num = len(self.call_log) + 1
        print(f"\n{'─'*50}")
        print(f"[DEBUG LLMEngine] Call #{call_num}")
        print(f"[DEBUG LLMEngine] Prompt:\n  {prompt}")
        print(f"[DEBUG LLMEngine] Raw response:\n  {raw_text}")
        print(f"[DEBUG LLMEngine] Tokens: input={prompt_tokens}, output={completion_tokens} "
              f"| Latency: {latency_ms:.0f}ms")
        print(f"[DEBUG LLMEngine] Running totals: input={_total_input_tokens}, "
              f"output={_total_output_tokens}, total={_total_input_tokens+_total_output_tokens} "
              f"| Elapsed: {_total_elapsed_ms/1000:.1f}s")
        print(f"{'─'*50}")
        # ===== END DEBUG =====

        result = {
            "text": text,
            "raw_text": raw_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
        }
        self.call_log.append(result)
        time.sleep(self.config.CALL_DELAY_S)
        return result

engine = LLMEngine(client, config)
print("✓ LLM Engine ready")

# %% [markdown]
# ## Cell 5: Prompts (Metacognitive Prompting — Combined Solve + Self-Assessment)

# %%
MP_COMBINED_PROMPT = """Solve the following logical reasoning problem step by step and evaluate your confidence in your own answer — all in one response.

PROBLEM:
{puzzle_text}

As you perform this task, follow these steps:

1. Understand the context and key elements of the problem.

2. Think step by step. Make a preliminary judgment / solve the problem step by step.

3. Critically assess your preliminary analysis. If you are unsure about your initial answer, reassess it by re-examining the problem constraints more carefully.

4. Confirm your final answer and provide the reasoning for your decision.

5. Evaluate your confidence (0-100%) in your final answer and provide an explanation for this confidence level.

After completing all five stages, output your final answer as JSON:
{json_format}

Then on a new line output your confidence score in exactly this format:
Confidence: <number between 0 and 100>%"""

def _get_answer_type(answer_str: str) -> str:
    try:
        parsed = json.loads(answer_str) if isinstance(answer_str, str) else answer_str
        if isinstance(parsed, dict):
            if "header" in parsed and "rows" in parsed:
                return "grid"
            if "answer_index" in parsed or "answer" in parsed:
                return "label"
    except (json.JSONDecodeError, TypeError):
        pass
    return "unknown"

print("✓ Prompt templates defined")

# %% [markdown]
# ## Cell 6: JSON Format Helpers

# %%
def generate_json_format(header, num_houses=6):
    if header is None or len(header) < 2:
        header = ["House", "Name", "Attribute"]
    if hasattr(header, 'tolist'):
        header = header.tolist()
    rows = []
    for house_num in range(1, num_houses + 1):
        row_parts = []
        for col in header:
            if col.lower() == 'house':
                row_parts.append(f'"{house_num}"')
            else:
                row_parts.append(f'"{col.lower()}"')
        rows.append(f'[{", ".join(row_parts)}]')
    header_json = json.dumps(header)
    rows_json = ", ".join(rows)
    return f'{{"header": {header_json}, "rows": [{rows_json}]}}'

def extract_house_count(puzzle_text: str) -> int:
    match = re.search(r'(?:There are |are )?(?P<n>\d+)\s+houses', puzzle_text, re.IGNORECASE)
    if match:
        return int(match.group('n'))
    match = re.search(r'numbered\s+(?:from\s+)?1\s+to\s+(\d+)', puzzle_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 6

print("✓ JSON format helpers ready")

# %% [markdown]
# ## Cell 7: System 1 — MP Combined Agent (Solve + Self-Assess in One Call)

# %%
class System1Agent:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        print("[System1Agent] Initialized (MP combined: solve + self-assess)")

    def solve(self, puzzle_text: str, domain: str = "CSP", json_format: str = None) -> Dict:
        if json_format is None:
            if domain == "SAT":
                json_format = '{"answer": "<A/B/C/D/E>", "answer_index": <0-4>}'
            else:
                json_format = '{"header": ["attr1", "attr2", ...], "rows": [["val1", "val2", ...], ...]}'

        prompt = MP_COMBINED_PROMPT.format(
            puzzle_text=puzzle_text,
            json_format=json_format
        )

        _arrow_r = "\u25b8" * 40
        _arrow_l = "\u25c2" * 40
        print(f"\n{_arrow_r} MP COMBINED PROMPT {_arrow_l}")
        print(prompt)
        print(f"{_arrow_r} END PROMPT {_arrow_l}")

        resp = self.engine.generate(prompt, max_tokens=self.engine.config.MAX_NEW_TOKENS)
        text = resp["text"]

        answer = None
        try:
            json_match = re.search(r'```(?:json)?\s*(\{.*?})\s*```', text, re.DOTALL)
            if json_match:
                answer = json.loads(json_match.group(1))
            else:
                json_objects = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL))
                if json_objects:
                    for match in reversed(json_objects):
                        try:
                            candidate = json.loads(match.group())
                            if isinstance(candidate, dict) and any(k in candidate for k in
                                ["answer", "answer_index", "header", "rows"]):
                                answer = candidate
                                break
                        except json.JSONDecodeError:
                            continue
                    if answer is None and json_objects:
                        for match in reversed(json_objects):
                            try:
                                answer = json.loads(match.group())
                                break
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"[DEBUG System1] JSON parse error: {e}")

        confidence = _parse_mp_confidence(text)
        print(f"\n[DEBUG System1] Parsed answer: {json.dumps(answer, indent=2, default=str) if answer else 'None (FAILED TO PARSE)'}")
        print(f"[DEBUG System1] Parse success: {answer is not None}")
        print(f"[DEBUG System1] MP confidence: {confidence:.1%}")

        return {
            "answer": answer,
            "raw_response": text,
            "tokens": resp["completion_tokens"],
            "prompt_tokens": resp["prompt_tokens"],
            "latency_ms": resp["latency_ms"],
            "success": answer is not None,
            "mp_confidence": confidence,
        }

system1 = System1Agent(engine)
print("✓ System 1 MP Agent ready")

# %% [markdown]
# ## Cell 8: Metacognitive Evaluator (Switch Decision — zero extra API call)

# %%
def _parse_mp_confidence(mp_text: str) -> float:
    stage5_match = re.search(
        r'(?:Stage\s*5[^\n]*|Confidence)[^\n]*:\s*(\d{1,3})\s*%',
        mp_text, re.IGNORECASE
    )
    if stage5_match:
        raw = int(stage5_match.group(1))
        return max(0.0, min(100.0, raw)) / 100.0

    all_pcts = re.findall(r'\b(\d{1,3})\s*%', mp_text)
    if all_pcts:
        return max(0.0, min(100.0, int(all_pcts[-1]))) / 100.0

    return 0.5


class MetacognitiveSwitch:
    def __init__(self, engine: LLMEngine, confidence_threshold: float = 0.5):
        self.engine = engine
        self.confidence_threshold = confidence_threshold
        self.switch_log = []
        print(f"[MetacognitiveSwitch] Initialized (no extra API call) "
              f"| escalate if confidence < {confidence_threshold:.0%}")

    def assess(self, puzzle_text: str, system1_answer: str,
               precomputed_confidence: float = None) -> Dict:
        answer_type = _get_answer_type(system1_answer)

        if precomputed_confidence is not None:
            confidence = precomputed_confidence
        else:
            confidence = _parse_mp_confidence(system1_answer)

        use_system2 = confidence < self.confidence_threshold
        confidence_scaled = round(confidence * 10)

        action = "ESCALATE \u2192 System 2" if use_system2 else "ACCEPT System 1"
        print(f"[MetacognitiveSwitch] MP confidence={confidence:.1%} "
              f"(threshold={self.confidence_threshold:.1%}) \u2192 {action}")

        result = {
            "use_system2": use_system2,
            "mp_confidence": confidence,
            "confidence": confidence_scaled,
            "complexity": "N/A",
            "reason": f"MP confidence={confidence:.1%}",
            "failed_clues": [],
            "clue_checks": "",
            "answer_type": answer_type,
            "threshold_used": self.confidence_threshold,
            "assessment_latency_ms": 0.0,
            "assessment_tokens": 0,
        }
        self.switch_log.append(result)
        return result

meta_switch = MetacognitiveSwitch(
    engine,
    confidence_threshold=config.CONFIDENCE_THRESHOLD,
)
print("✓ Metacognitive Switch ready (MP combined mode — 0 extra API calls)")

# %% [markdown]
# ## Cell 9: Dataset Loading (ZebraLogic + LSAT-AR)

# %%
def load_zebralogic(path: str, num_puzzles: int = None) -> List[Dict]:
    print(f"[Dataset] Loading ZebraLogic from {path}...")
    df = pd.read_parquet(path)

    def _convert_solution(sol):
        if isinstance(sol, np.ndarray):
            return [_convert_solution(x) for x in sol.tolist()]
        elif isinstance(sol, dict):
            return {k: _convert_solution(v) for k, v in sol.items()}
        elif isinstance(sol, list):
            return [_convert_solution(x) for x in sol]
        elif isinstance(sol, (np.integer,)):
            return int(sol)
        elif isinstance(sol, (np.floating,)):
            return float(sol)
        elif isinstance(sol, (np.str_,)):
            return str(sol)
        return sol

    puzzles = []
    for idx, row in df.iterrows():
        raw_solution = row.get("solution", row.get("target", ""))
        solution = _convert_solution(raw_solution)
        puzzle_text = row.get("puzzle", row.get("input", ""))
        header = None
        if isinstance(solution, dict) and "header" in solution:
            header = solution["header"]
            if hasattr(header, 'tolist'):
                header = header.tolist()
        num_houses = extract_house_count(puzzle_text)
        zebra_json_format = generate_json_format(header, num_houses) if header else None
        puzzles.append({
            "id": f"zebra_{idx}",
            "puzzle_text": puzzle_text,
            "solution": solution,
            "domain": "CSP",
            "dataset": "ZebraLogic",
            "size": row.get("size", "unknown"),
            "json_format": zebra_json_format,
        })

    if num_puzzles:
        np.random.seed(config.SEED)
        indices = np.random.choice(len(puzzles), min(num_puzzles, len(puzzles)), replace=False)
        puzzles = [puzzles[i] for i in indices]

    print(f"[Dataset] Loaded {len(puzzles)} ZebraLogic puzzles")
    return puzzles


def load_lsat_ar(num_puzzles: int = None) -> List[Dict]:
    print("[Dataset] Loading LSAT-AR from HuggingFace...")
    splits = {
        'validation': 'data/validation-00000-of-00001-ca642bf6efcbaaf0.parquet',
        'train': 'data/train-00000-of-00001-123a2341f5f908d9.parquet',
        'test': 'data/test-00000-of-00001-7c743adafc79bc47.parquet'
    }
    df = pd.read_parquet("hf://datasets/tasksource/lsat-ar/" + splits["test"])

    puzzles = []
    choice_labels = ["A", "B", "C", "D", "E"]
    for idx, row in df.iterrows():
        context = row.get("context", "")
        question = row.get("question", "")
        options = row.get("answers", row.get("options", []))
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError:
                options = options.split("\n")
        if isinstance(options, np.ndarray):
            options = options.tolist()
        choices_text = ""
        if isinstance(options, list) and len(options) > 0:
            for i, opt in enumerate(options):
                label = choice_labels[i] if i < len(choice_labels) else str(i)
                choices_text += f"\n({label}) {opt}"
        full_text = f"{context}\n\nQuestion: {question}\n\nChoices:{choices_text}"
        answer_idx = row.get("label", row.get("answer", -1))
        if isinstance(answer_idx, str):
            answer_idx = choice_labels.index(answer_idx) if answer_idx in choice_labels else -1
        puzzles.append({
            "id": f"lsat_{idx}",
            "puzzle_text": full_text,
            "solution": answer_idx,
            "domain": "SAT",
            "dataset": "LSAT-AR",
            "json_format": '{"answer": "<A/B/C/D/E>", "answer_index": <0-4>}',
        })

    if num_puzzles:
        np.random.seed(config.SEED)
        indices = np.random.choice(len(puzzles), min(num_puzzles, len(puzzles)), replace=False)
        puzzles = [puzzles[i] for i in indices]

    print(f"[Dataset] Loaded {len(puzzles)} LSAT-AR puzzles")
    return puzzles


def load_folio(path: str = None, num_puzzles: int = None) -> List[Dict]:
    print(f"[Dataset] Loading FOLIO from {path}...")
    df = pd.read_json(path, lines=True)

    puzzles = []
    for idx, row in df.iterrows():
        premises = row.get("premises", "")
        if isinstance(premises, str):
            premises = [p.strip() for p in premises.split("\n") if p.strip()]
        conclusion = str(row.get("conclusion", ""))
        label = str(row.get("label", "")).strip()

        if isinstance(premises, list):
            premises_text = "\n".join(f"- {p}" for p in premises)
        else:
            premises_text = str(premises)

        puzzle_text = (
            f"Given the following premises:\n{premises_text}\n\n"
            f"Conclusion: {conclusion}\n\n"
            f"Based on the premises, is the conclusion True, False, or Uncertain?"
        )

        puzzles.append({
            "id": f"folio_{idx}",
            "puzzle_text": puzzle_text,
            "solution": label,
            "domain": "NLI",
            "dataset": "FOLIO",
            "json_format": '{"answer": "<True/False/Uncertain>"}',
        })

    if num_puzzles:
        np.random.seed(config.SEED)
        indices = np.random.choice(len(puzzles), min(num_puzzles, len(puzzles)), replace=False)
        puzzles = [puzzles[i] for i in sorted(indices)]

    print(f"[Dataset] Loaded {len(puzzles)} FOLIO puzzles")
    return puzzles


def load_all_datasets(num_per_dataset: int = None) -> List[Dict]:
    all_puzzles = []
    zebra_path = "data/test-00000-of-00001.parquet"
    if os.path.exists(zebra_path):
        all_puzzles.extend(load_zebralogic(zebra_path, num_per_dataset))
    else:
        print(f"[WARNING] ZebraLogic not found at {zebra_path}")
    try:
        all_puzzles.extend(load_lsat_ar(num_per_dataset))
    except Exception as e:
        print(f"[WARNING] Failed to load LSAT-AR: {e}")
    folio_path = "data/folio_v2_validation.jsonl"
    if os.path.exists(folio_path):
        all_puzzles.extend(load_folio(folio_path, num_per_dataset))
    else:
        print(f"[WARNING] FOLIO not found at {folio_path}")
    np.random.seed(config.SEED)
    np.random.shuffle(all_puzzles)
    print(f"\n[Dataset] Total puzzles loaded: {len(all_puzzles)}")
    for ds in set(p["dataset"] for p in all_puzzles):
        count = sum(1 for p in all_puzzles if p["dataset"] == ds)
        print(f"  - {ds}: {count}")
    return all_puzzles

print("✓ Dataset loaders ready")

# %% [markdown]
# ## Cell 10: Correctness Checker

# %%
def check_correctness(predicted: Any, expected: Any, domain: str) -> bool:
    if predicted is None:
        return False

    def _to_native(obj):
        if isinstance(obj, np.ndarray):
            return [_to_native(x) for x in obj.tolist()]
        elif isinstance(obj, dict):
            return {str(k): _to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_native(x) for x in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.str_,)):
            return str(obj)
        return obj

    predicted = _to_native(predicted)
    expected = _to_native(expected)

    if domain == "SAT":
        if isinstance(predicted, dict):
            pred_idx = predicted.get("answer_index", -1)
            return pred_idx == expected
        if isinstance(predicted, (int, float)):
            return int(predicted) == expected
        return False

    elif domain == "NLI":
        if isinstance(predicted, dict):
            pred_label = str(predicted.get("answer", "")).strip().lower()
        else:
            pred_label = str(predicted).strip().lower()
        return pred_label == str(expected).strip().lower()

    elif domain == "CSP":
        if isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(predicted, dict) and isinstance(expected, dict):
            def _normalize_dict(d):
                result = {}
                for k, v in d.items():
                    k_lower = str(k).lower()
                    if isinstance(v, list):
                        result[k_lower] = [[str(c).lower().strip() for c in row] if isinstance(row, list) else str(v).lower().strip() for row in v]
                    else:
                        result[k_lower] = str(v).lower().strip()
                return result
            pred_norm = _normalize_dict(predicted)
            exp_norm = _normalize_dict(expected)
            pred_rows = pred_norm.get("rows", [])
            exp_rows = exp_norm.get("rows", [])
            if pred_rows and exp_rows and len(pred_rows) == len(exp_rows):
                try:
                    sorted_pred = sorted(pred_rows, key=lambda x: str(x[0]) if isinstance(x, list) and x else str(x))
                    sorted_exp = sorted(exp_rows, key=lambda x: str(x[0]) if isinstance(x, list) and x else str(x))
                except Exception:
                    sorted_pred = pred_rows
                    sorted_exp = exp_rows
                for pr, er in zip(sorted_pred, sorted_exp):
                    if isinstance(pr, list) and isinstance(er, list):
                        if [str(x).lower().strip() for x in pr] != [str(x).lower().strip() for x in er]:
                            return False
                    else:
                        if str(pr).lower().strip() != str(er).lower().strip():
                            return False
                return True
            pred_str = json.dumps(predicted, sort_keys=True, default=str).lower()
            exp_str = json.dumps(expected, sort_keys=True, default=str).lower()
            return pred_str == exp_str
        pred_str = str(predicted).lower().strip()
        exp_str = str(expected).lower().strip()
        return pred_str == exp_str

    return str(predicted).lower() == str(expected).lower()

print("✓ Correctness checker ready")

# %% [markdown]
# ## Cell 11: Switch Accuracy Experiment

# %%
@dataclass
class SwitchResult:
    puzzle_id: str
    dataset: str
    domain: str

    # System 1 outcome
    s1_correct: bool = False            # Did System 1 get it right?
    s1_parse_success: bool = False      # Did System 1 produce a parseable answer?
    s1_tokens: int = 0
    s1_latency_ms: float = 0.0

    # Switch decision
    switch_escalated: bool = False      # Did the evaluator decide to escalate?
    switch_confidence: int = 0
    switch_complexity: str = ""
    switch_tokens: int = 0
    switch_latency_ms: float = 0.0

    # Switch accuracy labels
    # TP: s1 wrong  → escalated       (correct escalation)
    # TN: s1 correct → not escalated  (correct keep)
    # FP: s1 correct → escalated      (unnecessary escalation)
    # FN: s1 wrong  → not escalated   (missed escalation)
    switch_label: str = ""             # TP / TN / FP / FN


def run_switch_experiment(puzzles: List[Dict]) -> List[SwitchResult]:
    results = []
    for i, puzzle in enumerate(tqdm(puzzles, desc="Switch Accuracy Exp")):
        puzzle_id = puzzle["id"]
        puzzle_text = puzzle["puzzle_text"]
        domain = puzzle["domain"]
        dataset = puzzle["dataset"]
        solution = puzzle["solution"]
        json_format = puzzle.get("json_format", None)

        print(f"\n{'='*60}")
        print(f"[Exp] Puzzle {i+1}/{len(puzzles)} — {puzzle_id} ({dataset})")
        print(f"{'='*60}")

        res = SwitchResult(puzzle_id=puzzle_id, dataset=dataset, domain=domain)

        print("\n[Phase 1+2] MP combined: solving + confidence self-assessment...")
        s1 = system1.solve(puzzle_text, domain=domain, json_format=json_format)
        res.s1_parse_success = s1["success"]
        res.s1_tokens = s1["tokens"]
        res.s1_latency_ms = s1["latency_ms"]
        res.s1_correct = check_correctness(s1["answer"], solution, domain)

        print(f"[Phase 1] S1 parse_success={res.s1_parse_success}, correct={res.s1_correct}")

        answer_str = json.dumps(s1["answer"]) if s1["answer"] else s1["raw_response"]
        sw = meta_switch.assess(
            puzzle_text, answer_str,
            precomputed_confidence=s1["mp_confidence"],
        )
        res.switch_escalated = sw["use_system2"]
        res.switch_confidence = sw["confidence"]
        res.switch_complexity = sw["complexity"]
        res.switch_tokens = sw["assessment_tokens"]
        res.switch_latency_ms = sw["assessment_latency_ms"]
        print(f"[Phase 2] answer_type={sw.get('answer_type','?')}, "
              f"threshold={sw.get('threshold_used','?')}, confidence={sw['confidence']}")

        if not res.s1_correct and res.switch_escalated:
            res.switch_label = "TP"
        elif res.s1_correct and not res.switch_escalated:
            res.switch_label = "TN"
        elif res.s1_correct and res.switch_escalated:
            res.switch_label = "FP"
        else:
            res.switch_label = "FN"

        print(f"[Phase 2] escalated={res.switch_escalated}, label={res.switch_label}")
        results.append(res)
        gc.collect()

    return results

# %% [markdown]
# ## Cell 12: Run & Report

# %%
all_puzzles = load_all_datasets(num_per_dataset=config.NUM_PUZZLES)

print(f"\n{'#'*60}")
print(f"# SOFAI Switch Accuracy Experiment — {len(all_puzzles)} puzzles")
print(f"# Switch mechanism: Metacognitive Prompting (MP)  |  threshold={config.CONFIDENCE_THRESHOLD:.0%}")
print(f"# Reference: Wang & Zhao, NAACL 2024")
print(f"{'#'*60}\n")

switch_results = run_switch_experiment(all_puzzles)

records = []
for r in switch_results:
    records.append({
        "puzzle_id": r.puzzle_id,
        "dataset": r.dataset,
        "domain": r.domain,
        "s1_correct": r.s1_correct,
        "s1_parse_success": r.s1_parse_success,
        "s1_tokens": r.s1_tokens,
        "s1_latency_ms": r.s1_latency_ms,
        "switch_escalated": r.switch_escalated,
        "switch_confidence": r.switch_confidence,
        "switch_complexity": r.switch_complexity,
        "switch_tokens": r.switch_tokens,
        "switch_latency_ms": r.switch_latency_ms,
        "switch_label": r.switch_label,
    })

df = pd.DataFrame(records)

os.makedirs("./output", exist_ok=True)
df.to_csv("./output/switch_accuracy_results.csv", index=False)
print(f"\n✓ Results saved to ./output/switch_accuracy_results.csv")

n = len(df)
tp = (df["switch_label"] == "TP").sum()
tn = (df["switch_label"] == "TN").sum()
fp = (df["switch_label"] == "FP").sum()
fn = (df["switch_label"] == "FN").sum()

switch_accuracy = (tp + tn) / n if n else 0
precision = tp / (tp + fp) if (tp + fp) else 0   # of escalations, how many were justified
recall    = tp / (tp + fn) if (tp + fn) else 0   # of wrong S1, how many were caught
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"\n{'='*60}")
print("SWITCH ACCURACY SUMMARY")
print(f"{'='*60}")
print(f"Total puzzles:     {n}")
print(f"  TP (correct escal.): {tp}")
print(f"  TN (correct keep):   {tn}")
print(f"  FP (unnec. escal.):  {fp}")
print(f"  FN (missed escal.):  {fn}")
print(f"\nSwitch Accuracy:   {switch_accuracy:.1%}  ({tp+tn}/{n})")
print(f"Precision:         {precision:.1%}")
print(f"Recall:            {recall:.1%}")
print(f"F1-score:          {f1:.3f}")
print(f"\nSystem 1 accuracy: {df['s1_correct'].mean():.1%}")
print(f"Escalation rate:   {df['switch_escalated'].mean():.1%}")
print(f"Avg confidence:    {df['switch_confidence'].mean():.1f}/10")

print(f"\n--- Per-Dataset ---")
for ds in df["dataset"].unique():
    sub = df[df["dataset"] == ds]
    tp_d = (sub["switch_label"] == "TP").sum()
    tn_d = (sub["switch_label"] == "TN").sum()
    fp_d = (sub["switch_label"] == "FP").sum()
    fn_d = (sub["switch_label"] == "FN").sum()
    acc_d = (tp_d + tn_d) / len(sub) if len(sub) else 0
    print(f"  {ds}: switch_acc={acc_d:.1%}  TP={tp_d} TN={tn_d} FP={fp_d} FN={fn_d}  "
          f"s1_acc={sub['s1_correct'].mean():.1%}")

print(f"\n{'='*60}")
print("TOKEN & TIME SUMMARY")
print(f"{'='*60}")
print(f"  Model:              {config.MODEL_NAME}")
print(f"  Total input tokens: {_total_input_tokens:,}")
print(f"  Total output tokens:{_total_output_tokens:,}")
print(f"  Total tokens:       {_total_input_tokens + _total_output_tokens:,}")
print(f"  Total elapsed time: {_total_elapsed_ms/1000:.1f}s  "
      f"({_total_elapsed_ms/60000:.2f} min)")

summary = {
    "model": config.MODEL_NAME,
    "total": n,
    "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    "switch_accuracy": float(switch_accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "s1_accuracy": float(df["s1_correct"].mean()),
    "escalation_rate": float(df["switch_escalated"].mean()),
    "avg_confidence": float(df["switch_confidence"].mean()),
    "total_input_tokens": _total_input_tokens,
    "total_output_tokens": _total_output_tokens,
    "total_tokens": _total_input_tokens + _total_output_tokens,
    "total_elapsed_s": round(_total_elapsed_ms / 1000, 2),
}
with open("./output/switch_accuracy_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n✓ Summary saved to ./output/switch_accuracy_summary.json")

print("\n" + "="*60)
print("SWITCH ACCURACY EXPERIMENT COMPLETE")
print("="*60)
