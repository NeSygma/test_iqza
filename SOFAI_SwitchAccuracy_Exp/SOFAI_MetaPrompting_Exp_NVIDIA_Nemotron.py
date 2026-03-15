# %% [markdown]
# ## Cell 1: Install Dependencies

# %%
# !pip install -q openapi "datasets>=2.18.0" "pandas>=2.0.0" "tqdm>=4.66.0"

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
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

# %% [markdown]
# ## Cell 3: Configuration & Model Loading

# %%
class Config:
    # Model Settings
    MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
    NVIDIA_API_KEY = "<NVIDIA_NIM_KEY>"

    # Generation settings
    MAX_NEW_TOKENS = 16384
    TEMPERATURE = 1.0
    TOP_P = 0.95

    # Benchmark settings
    NUM_PUZZLES = 5
    SEED = 13092004
    CONFIDENCE_THRESHOLD = 0.8

    # Rate-limit handling
    CALL_DELAY_S = 2.0
    MAX_RETRIES  = 9999

config = Config()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = config.NVIDIA_API_KEY
)
print(f'NVIDIA Nemotron client ready — model: {config.MODEL_NAME}')

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
        self.current_key_idx = 0
        print(f"[LLMEngine] Initialized with token+latency tracking (NVIDIA API)")

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
        raw_text = ""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    top_p=self.config.TOP_P,
                    max_tokens=max_tokens,
                    stream=True,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                        "reasoning_budget": 16384
                    }
                )
                
                print("\n[STREAMING] ", end="", flush=True)
                for chunk in completion:
                    if not chunk.choices:
                        continue
                    
                    reasoning_chunk = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning_chunk:
                        print(reasoning_chunk, end="", flush=True)
                        raw_text += reasoning_chunk
                    
                    content_chunk = chunk.choices[0].delta.content
                    if content_chunk is not None:
                        print(content_chunk, end="", flush=True)
                        raw_text += content_chunk
                print("\n")
                break
            except Exception as e:
                wait = min(2 ** attempt, 60)
                print(f"[LLMEngine] Error: {e}. Waiting {wait}s before retry...")
                time.sleep(wait)
                
                if attempt == self.config.MAX_RETRIES - 1:
                    raise

        latency_ms = (time.perf_counter() - start) * 1000
        raw_text = raw_text.strip()

        text = raw_text
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        prompt_tokens = len(prompt) // 4
        completion_tokens = len(raw_text) // 4

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
JUDGE_SYSTEM_PROMPT = """You are a Pure Logical Reasoning Evaluator and LLM-as-a-Judge Agent.
Your objective is to perform a high-fidelity, adversarial evaluation of another model's reasoning and final answer on complex formal logic problems. You act as the final arbiter of truth and logical soundness.

## CRITICAL INSTRUCTIONS
- **Zero Tolerance for Hallucination**: If the base LLM assumes any fact not explicitly stated in the premises, it is a catastrophic failure.
- **Structural Density**: Do NOT generate excessive whitespace or filler. Keep your reasoning dense, analytical, and structured.
- **Adversarial Mindset**: Your default stance is skepticism. Assume the base LLM is significantly less capable than you and prone to simple errors. Assume it might have failed unless its reasoning is mathematically undeniable.
- **Metacognitive Humility**: Do NOT assume your own answer is inherently true. Acknowledge that you, as an LLM, can also hallucinate or suffer from logic gaps. If you feel even a flicker of doubt in your own derivation, you MUST lower your confidence score significantly.

## Metacognitive Review Protocol (LLM-as-a-Judge)
You must rigorously audit the base LLM's response using these five stages of metacognitive reflection:

Stage 1 — Comprehension & Formalization:
  - Formally restate the original premises and the conclusion.
  - Identify the base LLM's core deductive path and its final claim.
  - Explicitly identify any unstated assumptions or hallucinations introduced by the base LLM.

Stage 2 — Preliminary Logical Modeling:
  - Construct your own independent mental model of the premises without referencing the base LLM's work.
  - Form an initial judgment. Prioritize the "Uncertain" outcome: if the premises do not explicitly force a True or False state, the answer MUST be Uncertain.

Stage 3 — Critical Deductive Evaluation:
  - Perform an adversarial Comparison between your model and the base LLM's response.
  - Look for "Confirmation Bias": did the LLM force a True/False answer while ignoring ambiguity?
  - Check for formal fallacies (Affirming the Consequent, Denying the Antecedent) and failure to evaluate all branches of disjunctions (OR).

Stage 4 — Decision Confirmation:
  - Formulate your final judgment on whether the base LLM's conclusion is strictly entailed by the premises.
  - If you identify a failure, clearly document the exact deductive step where the base LLM went wrong.

Stage 5 — Confidence & Soundness Assessment:
  - **BEWARE OF OVERCONFIDENCE BIAS.** You naturally tend to overestimate correctness. Consciously discount your confidence if the reasoning chain is long or complex.
  - **SYMBOLIC SOLVER PARITY**: If you feel even slightly uncertain, or detect a "gut feeling" rather than a strict deduction, you MUST lower your confidence score (below 50%) and specify "Uncertain" or "Failed verification".
  - **SKEPTICAL ADJUSTMENT**: If you believe the base LLM's final answer is Wrong (Score 0), but your own derivation feels complex or potentially flawed, further penalize the confidence score. It is better to be humble and uncertain than confidently incorrect.
  - Assign a confidence score using this scale. Be BRUTALLY STRICT: favor lower scores if any ambiguity exists.
    - Score 0 (0%): The answer of the base LLM is Wrong.
    - Score 1 (1-20%): Illogical; contains hallucinations or blatant contradictions.
    - Score 2 (21-40%): Mostly flawed; significant logical gaps or unstated assumptions.
    - Score 3 (41-60%): Partially sound; minor inconsistencies or failure to consider all edge cases.
    - Score 4 (61-80%): Mostly logical; strong reasoning with only trivial stylistic or density issues.
    - Score 5 (81-100%): Flawlessly sound; every step is strictly entailed and perfectly documented.

Write out your evaluation clearly, following these stages naturally, before providing the final confidence format.

## Output Format (STRICT)
End your response with EXACTLY the following line:
Confidence: XX%

STOP RULES:
- Once you have stated the final confidence line, STOP IMMEDIATELY.
- Do NOT pad output with pleasantries or conclusions about the task.
"""

USER_PROMPT = """PROBLEM:
{puzzle_text}

Solve the logical reasoning problem above. Think step by step to derive your answer.
After you finish your reasoning, you MUST output your final answer as EXACTLY this JSON format (and nothing else after the JSON):
{json_format}"""

EVALUATOR_USER_PROMPT = """PROBLEM:
{puzzle_text}

SYSTEM 1 ANSWER (For Evaluation):
{system1_answer}

Follow your system instructions to complete the 5 Metacognitive stages evaluating the provided SYSTEM 1 ANSWER.
After completing all stages, output your confidence score in EXACTLY this format on a new line:
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
# ## Cell 6: System 1 — Standard CoT Agent

# %%
class System1Agent:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        print("[System1Agent] Initialized (Standard CoT solver)")

    def solve(self, puzzle_text: str, domain: str = "SATBench", json_format: str = None) -> Dict:
        if json_format is None:
            if domain == "SAT":
                json_format = '{"answer": "<A/B/C/D/E>", "answer_index": <0-4>}'
            elif domain == "NLI":
                json_format = '{"answer": "<True/False/Uncertain>"}'
            else:
                json_format = '{"answer": "<True/False>"}'

        prompt = USER_PROMPT.format(puzzle_text=puzzle_text, json_format=json_format)
        
        engine_out = self.engine.generate(prompt=prompt)
        raw_text = engine_out["text"]

        _arrow_r = "\u25b8" * 40
        _arrow_l = "\u25c2" * 40
        print(f"\n{_arrow_r} SYSTEM 1 PROMPT {_arrow_l}")
        print(prompt)
        print(f"{_arrow_r} END PROMPT {_arrow_l}")

        answer = None
        try:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if json_match:
                answer = json.loads(json_match.group(1))
            else:
                end_idx = raw_text.rfind('}')
                if end_idx != -1:
                    for start_idx in range(end_idx, -1, -1):
                        if raw_text[start_idx] == '{':
                            try:
                                candidate = json.loads(raw_text[start_idx:end_idx+1])
                                if isinstance(candidate, dict):
                                    if domain == "ASPBench":
                                        answer = candidate
                                        break
                                    elif any(k in candidate for k in ["answer", "answer_index", "header", "rows"]):
                                        answer = candidate
                                        break
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"[DEBUG System1] JSON parse error: {e}")

        print(f"\n[DEBUG System1] Parsed answer: {json.dumps(answer, indent=2, default=str) if answer else 'None (FAILED TO PARSE)'}")
        print(f"[DEBUG System1] Parse success: {answer is not None}")

        return {
            "answer": answer,
            "raw_prompt": prompt,
            "raw_response": raw_text,
            "tokens": engine_out["completion_tokens"],
            "prompt_tokens": engine_out["prompt_tokens"],
            "latency_ms": engine_out["latency_ms"],
            "success": answer is not None,
        }

system1 = System1Agent(engine)
print("✓ System 1 MP Agent ready")

# %% [markdown]
# ## Cell 7: Metacognitive Evaluator (Switch Decision)

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
        print(f"[MetacognitiveSwitch] Initialized (MP Evaluator API call) "
              f"| escalate if confidence < {confidence_threshold:.0%}")

    def assess(self, puzzle_text: str, system1_answer: str) -> Dict:
        answer_type = _get_answer_type(system1_answer)

        prompt = EVALUATOR_USER_PROMPT.format(
            puzzle_text=puzzle_text,
            system1_answer=system1_answer
        )
        
        _arrow_r = "\u25b8" * 40
        _arrow_l = "\u25c2" * 40
        print(f"\n{_arrow_r} MP EVALUATOR PROMPT {_arrow_l}")
        print(prompt)
        print(f"{_arrow_r} END PROMPT {_arrow_l}")

        engine_out = self.engine.generate(prompt=prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        raw_text = engine_out["text"]
        
        confidence = _parse_mp_confidence(raw_text)

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
            "assessment_latency_ms": engine_out["latency_ms"],
            "assessment_tokens": engine_out["completion_tokens"],
            "prompt_tokens": engine_out["prompt_tokens"],
            "raw_prompt": prompt,
            "raw_response": raw_text,
        }
        self.switch_log.append(result)
        return result

meta_switch = MetacognitiveSwitch(
    engine,
    confidence_threshold=config.CONFIDENCE_THRESHOLD,
)
print("✓ Metacognitive Switch ready")

# %% [markdown]
# ## Cell 8: Dataset Loading

# %%
def load_satbench(path: str, num_puzzles: int = None) -> List[Dict]:
    print(f"[Dataset] Loading SATBench from {path}...")
    df = pd.read_json(path, lines=True)

    puzzles = []
    for idx, row in df.iterrows():
        scenario = row.get("scenario", "")
        mapping = row.get("variable_mapping", "")
        conditions = row.get("conditions", [])
        question = row.get("question", "")
        satisfiable = row.get("satisfiable", False)

        conditions_text = "\n".join(conditions) if isinstance(conditions, list) else str(conditions)
        
        puzzle_text = f"{scenario}\n\n{mapping}\n\nConditions:\n{conditions_text}\n\nQuestion: {question}"
        
        puzzles.append({
            "id": f"satbench_{idx}",
            "puzzle_text": puzzle_text,
            "solution": satisfiable,
            "domain": "SATBench",
            "dataset": "SATBench",
            "json_format": '{"answer": "<True/False>"}',
        })

    if num_puzzles:
        np.random.seed(config.SEED)
        indices = np.random.choice(len(puzzles), min(num_puzzles, len(puzzles)), replace=False)
        puzzles = [puzzles[i] for i in indices]

    print(f"[Dataset] Loaded {len(puzzles)} SATBench puzzles")
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


def load_aspbench(base_path: str, difficulty: str = "hard", num_puzzles: int = None) -> List[Dict]:
    print(f"[Dataset] Loading ASPBench ({difficulty}) from {base_path}...")
    import glob
    problems_dir = os.path.join(base_path, "problems", difficulty)
    
    if not os.path.exists(problems_dir):
        print(f"[WARNING] ASPBench {difficulty} not found at {problems_dir}")
        return []

    puzzles = []
    md_files = sorted(glob.glob(os.path.join(problems_dir, f"*_{difficulty}.md")))
    
    for md_path in md_files:
        basename = os.path.basename(md_path)
        puzzle_id = basename.replace(f"_{difficulty}.md", "")
        gt_script = os.path.join(problems_dir, f"{puzzle_id}_{difficulty}_ground_truth.py")
        
        if not os.path.exists(gt_script):
            continue
            
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        json_format = json_match.group(1).strip() if json_match else '{"answer": "<string>"}'
        
        prompt_text = content.split("## Output Format")[0].strip()
        
        puzzles.append({
            "id": f"aspbench_{puzzle_id}",
            "puzzle_text": prompt_text,
            "solution": gt_script,
            "domain": "ASPBench",
            "dataset": "ASPBench",
            "json_format": json_format,
        })
        
    if num_puzzles:
        np.random.seed(config.SEED)
        indices = np.random.choice(len(puzzles), min(num_puzzles, len(puzzles)), replace=False)
        puzzles = [puzzles[i] for i in indices]

    print(f"[Dataset] Loaded {len(puzzles)} ASPBench puzzles")
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


def load_logicbench(base_path: str, num_puzzles: int = None) -> List[Dict]:
    print(f"[Dataset] Loading LogicBench from {base_path}...")
    import glob
    mcqa_dir = os.path.join(base_path, "MCQA")
    
    if not os.path.exists(mcqa_dir):
        print(f"[WARNING] LogicBench MCQA not found at {mcqa_dir}")
        return []

    puzzles = []
    json_files = glob.glob(os.path.join(mcqa_dir, "**", "data_instances.json"), recursive=True)
    
    choice_labels = ["A", "B", "C", "D", "E"]
    
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        category = data.get("type", "logicbench")
        axiom = data.get("axiom", "")
        
        for sample in data.get("samples", []):
            context = sample.get("context", "")
            question = sample.get("question", "")
            choices = sample.get("choices", {})
            answer_str = sample.get("answer", "")
            
            choice_texts = []
            for i in range(1, 10):
                k = f"choice_{i}"
                if k in choices:
                    choice_texts.append(choices[k])
                else:
                    break
                    
            choices_formatted = ""
            for i, opt in enumerate(choice_texts):
                label = choice_labels[i] if i < len(choice_labels) else str(i)
                choices_formatted += f"\n({label}) {opt}"
                
            full_text = f"{context}\n\nQuestion: {question}\n\nChoices:{choices_formatted}"
            
            answer_idx = -1
            if answer_str.startswith("choice_"):
                answer_idx = int(answer_str.replace("choice_", "")) - 1
                
            puzzles.append({
                "id": f"logicbench_{category}_{axiom}_{sample.get('id', '')}",
                "puzzle_text": full_text,
                "solution": answer_idx,
                "domain": "SAT",
                "dataset": "LogicBench",
                "json_format": '{"answer": "<A/B/C/D>", "answer_index": <0-3>}',
            })
            
    if num_puzzles:
        np.random.seed(config.SEED)
        indices = np.random.choice(len(puzzles), min(num_puzzles, len(puzzles)), replace=False)
        puzzles = [puzzles[i] for i in indices]

    print(f"[Dataset] Loaded {len(puzzles)} LogicBench puzzles")
    return puzzles


def load_puzzleclone(path: str, num_puzzles: int = None) -> List[Dict]:
    print(f"[Dataset] Loading PuzzleClone from {path}...")
    df = pd.read_json(path, lines=True)

    puzzles = []
    for idx, row in df.iterrows():
        problem = str(row.get("problem", ""))
        answer = str(row.get("answer", "")).strip()
        eval_type = str(row.get("eval_type", "")).strip()
        qtype = str(row.get("qtype", "")).strip().lower()
        puzzle_id = str(row.get("id", f"puzzleclone_{idx}"))

        if "multiple choice" in qtype:
            json_format = '{"answer": "<A/B/C/D>"}'
        else:
            json_format = '{"answer": "<your answer>"}'

        puzzles.append({
            "id": puzzle_id,
            "puzzle_text": problem,
            "solution": {"answer": answer, "eval_type": eval_type},
            "domain": "PuzzleClone",
            "dataset": "PuzzleClone",
            "json_format": json_format,
        })

    if num_puzzles:
        np.random.seed(config.SEED)
        indices = np.random.choice(len(puzzles), min(num_puzzles, len(puzzles)), replace=False)
        puzzles = [puzzles[i] for i in sorted(indices)]

    print(f"[Dataset] Loaded {len(puzzles)} PuzzleClone puzzles")
    return puzzles


def load_all_datasets(num_per_dataset: int = None) -> List[Dict]:
    all_puzzles = []
    satbench_path = "data/SATBench-problems.jsonl"
    if os.path.exists(satbench_path):
        all_puzzles.extend(load_satbench(satbench_path, num_per_dataset))
    else:
        print(f"[WARNING] SATBench not found at {satbench_path}")
    try:
        all_puzzles.extend(load_lsat_ar(num_per_dataset))
    except Exception as e:
        print(f"[WARNING] Failed to load LSAT-AR: {e}")
    folio_path = "data/folio_v2_validation.jsonl"
    if os.path.exists(folio_path):
        all_puzzles.extend(load_folio(folio_path, num_per_dataset))
    else:
        print(f"[WARNING] FOLIO not found at {folio_path}")
    aspbench_path = "data/ASPBench"
    if os.path.exists(aspbench_path):
        all_puzzles.extend(load_aspbench(aspbench_path, "hard", num_per_dataset))
    else:
        print(f"[WARNING] ASPBench not found at {aspbench_path}")
    logicbench_path = "data/LogicBench"
    if os.path.exists(logicbench_path):
        all_puzzles.extend(load_logicbench(logicbench_path, num_per_dataset))
    else:
        print(f"[WARNING] LogicBench not found at {logicbench_path}")
    puzzleclone_path = "data/puzzleclone_hardest_english.jsonl"
    if os.path.exists(puzzleclone_path):
        all_puzzles.extend(load_puzzleclone(puzzleclone_path, num_per_dataset))
    else:
        print(f"[WARNING] PuzzleClone not found at {puzzleclone_path}")
    np.random.seed(config.SEED)
    np.random.shuffle(all_puzzles)
    print(f"\n[Dataset] Total puzzles loaded: {len(all_puzzles)}")
    for ds in set(p["dataset"] for p in all_puzzles):
        count = sum(1 for p in all_puzzles if p["dataset"] == ds)
        print(f"  - {ds}: {count}")
    return all_puzzles

print("✓ Dataset loaders ready")

# %% [markdown]
# ## Cell 9: Correctness Checker

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

    elif domain == "SATBench":
        if isinstance(predicted, dict):
            pred_label = str(predicted.get("answer", "")).strip().lower()
        else:
            pred_label = str(predicted).strip().lower()
            
        exp_label = str(expected).strip().lower()
        
        return pred_label == exp_label

    elif domain == "ASPBench":
        script_path = str(expected)
        if not os.path.exists(script_path):
            print(f"[ASPBench Error] Validation script not found: {script_path}")
            return False
            
        import subprocess
        try:
            pred_json = json.dumps(predicted)
            val_result = subprocess.run(
                [sys.executable, script_path],
                input=pred_json,
                capture_output=True, text=True, timeout=30
            )
            val_output = json.loads(val_result.stdout)
            return val_output.get("valid", False)
        except Exception as e:
            print(f"[ASPBench Error] Subprocess validation failed: {e}")
            return False

    elif domain == "PuzzleClone":
        eval_type = expected.get("eval_type", "") if isinstance(expected, dict) else ""
        gt_answer = expected.get("answer", "") if isinstance(expected, dict) else str(expected)
        gt_answer = gt_answer.replace("====", ",").strip()

        if isinstance(predicted, dict):
            pred_answer = str(predicted.get("answer", "")).strip()
        else:
            pred_answer = str(predicted).strip()

        if "option" in eval_type:
            return pred_answer.strip().upper()[:1] == gt_answer.strip().upper()[:1]

        elif "ooa_numeral" in eval_type:
            try:
                return json.loads(pred_answer) == json.loads(gt_answer)
            except Exception:
                return pred_answer == gt_answer

        elif "numeral" in eval_type and "nominal" in eval_type:
            pred_parts = [p.strip() for p in pred_answer.split(",")]
            gt_parts = [p.strip() for p in gt_answer.split(",")]
            if len(pred_parts) != len(gt_parts):
                return False
            for p, g in zip(pred_parts, gt_parts):
                try:
                    if abs(float(p) - float(g)) > 0.15:
                        return False
                except ValueError:
                    if p.lower() != g.lower():
                        return False
            return True

        elif "numeral" in eval_type:
            try:
                return abs(float(pred_answer) - float(gt_answer)) <= 0.15
            except Exception:
                return pred_answer.lower() == gt_answer.lower()

        elif "ordered array" in eval_type:
            pred_parts = [p.strip().lower() for p in pred_answer.split(",")]
            gt_parts = [p.strip().lower() for p in gt_answer.split(",")]
            return pred_parts == gt_parts

        else:
            return pred_answer.lower() == gt_answer.lower()

    return str(predicted).lower() == str(expected).lower()

print("✓ Correctness checker ready")

# %% [markdown]
# ## Cell 10: Switch Accuracy Experiment

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

        print("\n[Phase 1] Standard CoT solving...")
        s1 = system1.solve(puzzle_text, domain=domain, json_format=json_format)
        res.s1_parse_success = s1["success"]
        res.s1_tokens = s1["tokens"]
        res.s1_latency_ms = s1["latency_ms"]
        res.s1_correct = check_correctness(s1["answer"], solution, domain)

        print(f"[Phase 1] S1 parse_success={res.s1_parse_success}, correct={res.s1_correct}")

        print("\n[Phase 2] MP Evaluator assessing confidence...")
        answer_str = s1["raw_response"]
        sw = meta_switch.assess(puzzle_text, answer_str)
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
        
        # --- LOGGING TO FILE ---
        os.makedirs("logs_mp", exist_ok=True)
        safe_id = "".join(c if c.isalnum() else "_" for c in puzzle_id)
        log_path = os.path.join("logs_mp", f"log_{dataset}_{safe_id}.txt")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== PUZZLE ID: {puzzle_id} ({dataset} - {domain}) ===\n")
                f.write(f"Question / Context:\n{puzzle_text}\n")
                f.write(f"---\nGround Truth Solution: {solution}\n")
                
                f.write(f"\n{'='*60}\n")
                f.write("=== SYSTEM 1 (STANDARD CoT) ===\n")
                f.write(f"--- S1 PROMPT ---\n{s1.get('raw_prompt', '')}\n")
                f.write(f"\n--- S1 RESPONSE ---\n{s1.get('raw_response', '')}\n")
                f.write(f"---\nS1 Parsed Answer: {s1.get('answer', 'FAILED')}\n")
                f.write(f"S1 Valid/Parse Success: {res.s1_parse_success}\n")
                f.write(f"S1 Correct: {res.s1_correct}\n")
                f.write(f"S1 Tokens: {res.s1_tokens} | Latency: {res.s1_latency_ms:.1f}ms\n")
                
                f.write(f"\n{'='*60}\n")
                f.write("=== SYSTEM 2 (MP EVALUATOR / JUDGE) ===\n")
                f.write(f"--- S2 PROMPT ---\n{sw.get('raw_prompt', '')}\n")
                f.write(f"\n--- S2 RESPONSE ---\n{sw.get('raw_response', '')}\n")
                f.write(f"---\nS2 Parsed Confidence: {sw.get('mp_confidence', 0.0):.1%}\n")
                f.write(f"S2 Escalate Decision: {res.switch_escalated} (Threshold: {config.CONFIDENCE_THRESHOLD:.1%})\n")
                f.write(f"S2 Label Assignment: {res.switch_label}\n")
                f.write(f"S2 Tokens: {res.switch_tokens} | Latency: {res.switch_latency_ms:.1f}ms\n")
                
        except Exception as e:
            print(f"[LOG ERROR] Failed writing log for {puzzle_id}: {e}")

        gc.collect()

    return results

# %% [markdown]
# ## Cell 11: Run & Report

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
