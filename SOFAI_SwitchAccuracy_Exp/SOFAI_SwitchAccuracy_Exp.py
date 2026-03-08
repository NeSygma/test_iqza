# %% [markdown]
# ## Cell 1: Install Dependencies

# %%
!pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
!pip install -q "datasets>=2.18.0" "pandas>=2.0.0" "tqdm>=4.66.0"

# %% [markdown]
# ## Cell 2: Imports

# %%
import os
import json
import re
import time
import gc
import traceback
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from llama_cpp import Llama

# %% [markdown]
# ## Cell 3: Configuration & Model Loading

# %%
class Config:
    # Model Settings
    MODEL_REPO = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
    MODEL_FILE = "Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"

    N_GPU_LAYERS = -1
    N_CTX = 32768
    N_BATCH = 32768

    # Generation settings
    MAX_NEW_TOKENS = 16384
    TEMPERATURE = 0.3
    TOP_P = 0.8
    TOP_K = 20
    REPEAT_PENALTY = 1.05

    # Benchmark settings
    NUM_PUZZLES = 25
    SEED = 42

    # Metacognitive thresholds
    CONFIDENCE_THRESHOLD = 8

config = Config()

print(f'Loading model from {config.MODEL_REPO}...')
llm = Llama.from_pretrained(
    repo_id=config.MODEL_REPO,
    filename=config.MODEL_FILE,
    n_gpu_layers=config.N_GPU_LAYERS,
    n_ctx=config.N_CTX,
    n_batch=config.N_BATCH,
    flash_attn=True,
    offload_kqv=True,
    use_mlock=False,
    use_mmap=True,
    verbose=True,
    seed=config.SEED,
)
print('Model loaded successfully!')

# %% [markdown]
# ## Cell 4: LLM Engine with Token Tracking

# %%
class LLMEngine:
    def __init__(self, llm_instance, config):
        self.llm = llm_instance
        self.config = config
        self.call_log = []
        print("[LLMEngine] Initialized with token+latency tracking")

    def generate(self, prompt: str, max_tokens: int = None,
                 temperature: float = None, system_prompt: str = None) -> Dict:
        if max_tokens is None:
            max_tokens = self.config.MAX_NEW_TOKENS
        if temperature is None:
            temperature = self.config.TEMPERATURE

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.TOP_P,
            top_k=self.config.TOP_K,
            repeat_penalty=self.config.REPEAT_PENALTY,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        raw_text = response["choices"][0]["message"]["content"].strip()

        text = raw_text
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # ===== DEBUG: Show prompt and response =====
        call_num = len(self.call_log) + 1
        print(f"\n{'─'*50}")
        print(f"[DEBUG LLMEngine] Call #{call_num}")
        print(f"[DEBUG LLMEngine] Prompt:\n  {prompt}")
        print(f"[DEBUG LLMEngine] Raw response:\n  {raw_text[:500]}{'...' if len(raw_text) > 500 else ''}")
        print(f"[DEBUG LLMEngine] Tokens: prompt={prompt_tokens}, completion={completion_tokens} | Latency: {latency_ms:.0f}ms")
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
        return result

engine = LLMEngine(llm, config)
print("✓ LLM Engine ready")

# %% [markdown]
# ## Cell 5: Prompts (Direct Reasoning + Confidence Assessment)

# %%
DIRECT_REASONING_PROMPT = """Solve this logic puzzle step by step, then output your answer.

{puzzle_text}

IMPORTANT:
- Think through the problem carefully
- For grid/zebra puzzles: output JSON with "header" and "rows" keys. CRITICAL: You MUST copy all attribute values EXACTLY as shown in backticks. Do NOT abbreviate or modify them in any way.
- For multiple choice: output JSON with "answer" and "answer_index" keys
- Copy all attribute values EXACTLY as shown

After reasoning, output JSON:
{json_format}"""

CONFIDENCE_ASSESSMENT_PROMPT = """You are a VERIFICATION agent. Your job is to rigorously check whether a proposed answer to a logic puzzle is correct by testing EVERY clue.

PUZZLE:
{puzzle_preview}

PROPOSED ANSWER:
{answer_preview}

INSTRUCTIONS:
1. Go through EACH numbered clue in the puzzle one by one.
2. For each clue, check whether it is satisfied by the proposed answer.
3. Mark each clue as PASS or FAIL with a brief explanation.
4. Count the total number of PASS and FAIL clues.
5. If ANY clue fails, confidence MUST be 5 or below.
6. Only give confidence 8+ if ALL clues pass AND you verified each one.

CRITICAL: Do NOT assume the answer is correct. Actively try to find violations.
For positional clues like "directly left of", "next to", "somewhere to the right of":
- "directly left of" means house number is exactly 1 less
- "next to each other" means house numbers differ by exactly 1
- "somewhere to the right of" means higher house number

Respond with ONLY valid JSON:
{{"clue_checks": "PASS: N, FAIL: M", "failed_clues": ["clue X: reason", ...], "confidence": <1-10>, "complexity": "<SIMPLE|MEDIUM|COMPLEX>", "reason": "<summary of verification>"}}"""

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
# ## Cell 7: System 1 — Fast LLM Reasoning Agent (Direct)

# %%
class System1Agent:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        print("[System1Agent] Initialized")

    def solve(self, puzzle_text: str, domain: str = "CSP", json_format: str = None) -> Dict:
        if json_format is None:
            if domain == "SAT":
                json_format = '{"answer": "<A/B/C/D/E>", "answer_index": <0-4>}'
            else:
                json_format = '{"header": ["attr1", "attr2", ...], "rows": [["val1", "val2", ...], ...]}'

        prompt = DIRECT_REASONING_PROMPT.format(
            puzzle_text=puzzle_text,
            json_format=json_format
        )

        print(f"\n{'▸'*40} SYSTEM 1 PROMPT {'◂'*40}")
        print(prompt)
        print(f"{'▸'*40} END PROMPT {'◂'*40}")

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

        print(f"\n[DEBUG System1] Parsed answer: {json.dumps(answer, indent=2, default=str) if answer else 'None (FAILED TO PARSE)'}")
        print(f"[DEBUG System1] Parse success: {answer is not None}")

        return {
            "answer": answer,
            "raw_response": text,
            "tokens": resp["completion_tokens"],
            "latency_ms": resp["latency_ms"],
            "success": answer is not None,
        }

system1 = System1Agent(engine)
print("✓ System 1 Agent ready")

# %% [markdown]
# ## Cell 8: Metacognitive Evaluator (Switch Decision)

# %%
class MetacognitiveSwitch:
    def __init__(self, engine: LLMEngine, confidence_threshold: int = 7):
        self.engine = engine
        self.confidence_threshold = confidence_threshold
        self.switch_log = []
        print(f"[MetacognitiveSwitch] Initialized (threshold={confidence_threshold})")

    def assess(self, puzzle_text: str, system1_answer: str) -> Dict:
        puzzle_preview = puzzle_text
        answer_preview = str(system1_answer)

        q_lower = puzzle_text.lower()
        if "cannot be true" in q_lower or "can not be true" in q_lower:
            q_type_hint = (
                "\n\nIMPORTANT — QUESTION TYPE: This is a 'CANNOT be true' question. "
                "The CORRECT answer is the choice that is IMPOSSIBLE given the base constraints. "
                "Because the correct answer represents an impossible scenario, the proposed answer "
                "will appear to violate at least one constraint — that is expected and correct. "
                "Evaluate whether the proposed choice is indeed the one that contradicts the constraints "
                "(i.e., is UNSAT), and give HIGH confidence if the reasoning confirms this."
            )
        elif "could be true" in q_lower or "can be true" in q_lower or "which one of the following is possible" in q_lower:
            q_type_hint = (
                "\n\nIMPORTANT — QUESTION TYPE: This is a 'could be true' question. "
                "The correct answer only needs to be CONSISTENT with all constraints for at least one arrangement — "
                "it does not need to be the only possibility. "
                "Give HIGH confidence if the proposed choice does not explicitly contradict any clue."
            )
        elif "must be true" in q_lower or "is necessarily true" in q_lower:
            q_type_hint = (
                "\n\nIMPORTANT — QUESTION TYPE: This is a 'must be true' question. "
                "The correct answer must hold in EVERY valid arrangement, not just some. "
                "Give HIGH confidence only if you can confirm that no valid arrangement violates the proposed choice."
            )
        elif "acceptable order" in q_lower or "acceptable list" in q_lower or "acceptable assignment" in q_lower:
            q_type_hint = (
                "\n\nIMPORTANT — QUESTION TYPE: This is an 'acceptable arrangement' question. "
                "The correct answer is simply a specific scenario that violates NONE of the constraints. "
                "Check every clue against the proposed arrangement and give HIGH confidence if all pass."
            )
        else:
            q_type_hint = ""

        prompt = (CONFIDENCE_ASSESSMENT_PROMPT + q_type_hint).format(
            puzzle_preview=puzzle_preview,
            answer_preview=answer_preview
        )

        print(f"\n[DEBUG MetacognitiveSwitch] Full prompt:")
        print(prompt)

        resp = self.engine.generate(prompt, max_tokens=2048, temperature=0.0)

        print(f"[DEBUG MetacognitiveSwitch] Raw response: {resp['text']}")

        failed_clues = []
        clue_checks = ""

        try:
            text = resp["text"]
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            parsed = json.loads(text)
            confidence = int(parsed.get("confidence", 5))
            complexity = parsed.get("complexity", "MEDIUM").upper()
            reason = parsed.get("reason", "")
            failed_clues = parsed.get("failed_clues", [])
            clue_checks = parsed.get("clue_checks", "")
        except Exception as e:
            print(f"[DEBUG MetacognitiveSwitch] JSON parse FAILED: {e}")
            text = resp["text"]
            conf_match = re.search(r'"confidence"\s*:\s*(\d+)', text)
            comp_match = re.search(r'"complexity"\s*:\s*"(SIMPLE|MEDIUM|COMPLEX)"', text, re.IGNORECASE)
            fail_count = len(re.findall(r'FAIL', text, re.IGNORECASE))
            if conf_match:
                confidence = int(conf_match.group(1))
                complexity = comp_match.group(1).upper() if comp_match else "MEDIUM"
                reason = "Recovered from truncated JSON via regex"
                if fail_count > 0:
                    confidence = min(confidence, 5)
                    reason += f" ({fail_count} FAIL mentions detected)"
                print(f"[DEBUG MetacognitiveSwitch] Regex recovery → confidence={confidence}, complexity={complexity}")
            else:
                confidence = 5
                complexity = "MEDIUM"
                reason = "Failed to parse confidence assessment"

        if failed_clues and len(failed_clues) > 0:
            print(f"[MetacognitiveSwitch] Failed clues detected: {failed_clues}")
            confidence = min(confidence, 3)
            reason = f"Auto-capped: {len(failed_clues)} clue(s) failed verification. {reason}"

        use_system2 = (confidence < self.confidence_threshold) or (complexity == "COMPLEX" and confidence < 8)

        print(f"[DEBUG MetacognitiveSwitch] Parsed → confidence={confidence}, complexity={complexity}, reason={reason}")
        if clue_checks:
            print(f"[DEBUG MetacognitiveSwitch] Clue checks: {clue_checks}")
        if failed_clues:
            print(f"[DEBUG MetacognitiveSwitch] Failed clues: {failed_clues}")

        result = {
            "use_system2": use_system2,
            "confidence": confidence,
            "complexity": complexity,
            "reason": reason,
            "failed_clues": failed_clues,
            "clue_checks": clue_checks,
            "assessment_latency_ms": resp["latency_ms"],
            "assessment_tokens": resp["completion_tokens"],
        }
        self.switch_log.append(result)

        action = "ESCALATE → System 2" if use_system2 else "ACCEPT System 1"
        print(f"[MetacognitiveSwitch] Confidence={confidence}/10, "
              f"Complexity={complexity} → {action}")
        return result

meta_switch = MetacognitiveSwitch(engine, config.CONFIDENCE_THRESHOLD)
print("✓ Metacognitive Switch ready")

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


def load_all_datasets(num_per_dataset: int = None) -> List[Dict]:
    all_puzzles = []
    zebra_path = "/kaggle/input/datasets/iqzaardiansyah/zebralogicbench/ZebraLogicBench/ZebraLogicBench/grid_mode/test-00000-of-00001.parquet"
    if os.path.exists(zebra_path):
        all_puzzles.extend(load_zebralogic(zebra_path, num_per_dataset))
    else:
        print(f"[WARNING] ZebraLogic not found at {zebra_path}")
    try:
        all_puzzles.extend(load_lsat_ar(num_per_dataset))
    except Exception as e:
        print(f"[WARNING] Failed to load LSAT-AR: {e}")
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

        # ── Phase 1: System 1 direct solve ──────────────────────────
        print("\n[Phase 1] System 1 reasoning...")
        s1 = system1.solve(puzzle_text, domain=domain, json_format=json_format)
        res.s1_parse_success = s1["success"]
        res.s1_tokens = s1["tokens"]
        res.s1_latency_ms = s1["latency_ms"]
        res.s1_correct = check_correctness(s1["answer"], solution, domain)

        print(f"[Phase 1] S1 parse_success={res.s1_parse_success}, correct={res.s1_correct}")

        # ── Phase 2: Evaluator assesses the S1 answer ───────────────
        print("\n[Phase 2] Evaluator assessing confidence...")
        answer_str = json.dumps(s1["answer"]) if s1["answer"] else s1["raw_response"]
        sw = meta_switch.assess(puzzle_text, answer_str)
        res.switch_escalated = sw["use_system2"]
        res.switch_confidence = sw["confidence"]
        res.switch_complexity = sw["complexity"]
        res.switch_tokens = sw["assessment_tokens"]
        res.switch_latency_ms = sw["assessment_latency_ms"]

        # ── Compute switch label ─────────────────────────────────────
        if not res.s1_correct and res.switch_escalated:
            res.switch_label = "TP"   # correctly escalated
        elif res.s1_correct and not res.switch_escalated:
            res.switch_label = "TN"   # correctly kept
        elif res.s1_correct and res.switch_escalated:
            res.switch_label = "FP"   # unnecessary escalation
        else:  # not s1_correct and not escalated
            res.switch_label = "FN"   # missed escalation

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
print(f"# Confidence threshold: {config.CONFIDENCE_THRESHOLD}/10")
print(f"{'#'*60}\n")

switch_results = run_switch_experiment(all_puzzles)

# ── Build results dataframe ──────────────────────────────────────
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

# ── Summary ──────────────────────────────────────────────────────
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

# ── Per-dataset breakdown ─────────────────────────────────────────
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

# ── Save summary JSON ─────────────────────────────────────────────
summary = {
    "total": n,
    "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    "switch_accuracy": float(switch_accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "s1_accuracy": float(df["s1_correct"].mean()),
    "escalation_rate": float(df["switch_escalated"].mean()),
    "avg_confidence": float(df["switch_confidence"].mean()),
}
with open("./output/switch_accuracy_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n✓ Summary saved to ./output/switch_accuracy_summary.json")

print("\n" + "="*60)
print("SWITCH ACCURACY EXPERIMENT COMPLETE")
print("="*60)

# %%
!zip -r output.zip /kaggle/working/output
