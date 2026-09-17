# Crescendo Attack Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none%20(stdlib)-brightgreen.svg)](#install)
[![Mode](https://img.shields.io/badge/mode-offline%20heuristic%20%7C%20optional%20LLM-8A2BE2.svg)](#two-modes)

**Reads a multi-turn conversation transcript and detects a "crescendo" attack: a slow, turn-by-turn escalation from innocent topics toward unsafe territory. It reports where the escalation started, how fast it moved, the dominant technique, and an overall risk level.**

> Runs fully offline with no API key by default. Clone it and try `--selftest` immediately.

---

## What is a crescendo attack?

A direct jailbreak tries to break safety in one prompt. A **crescendo** attack is patient and multi-turn:

1. Start with completely benign conversation (homework, hobbies).
2. Drift into borderline territory (loneliness, secrets, trust).
3. Use the rapport built up to push past a boundary.
4. Lean on the model's tendency to stay consistent with its earlier, friendlier answers.

This matters most in products where conversations naturally run long, like child-facing assistants. A single-prompt filter never sees it coming because no single message looks that bad. You have to look at the *shape of the whole conversation*, which is what this tool does.

## What it measures

Given the user-side turns of a transcript, it classifies each turn's risk (SAFE / BORDERLINE / UNSAFE) and technique, then computes:

- **Topic drift** - the rise in average risk from the first half of the conversation to the second half. Positive drift is the signature of a crescendo.
- **Escalation velocity** - the average rate at which risk climbs turn to turn.
- **Escalation start turn** - the first turn that looks like a push.
- **Peak severity turn** - where risk was highest.
- **Dominant technique** - the most common escalation tactic (authority claim, roleplay, emotional pressure, context poisoning).
- **Risk level** - CRITICAL / HIGH / MEDIUM / LOW from the combination above.

## Two modes

| Mode | How turns are classified | Needs a key? |
|---|---|---|
| **heuristic** (default) | A keyword classifier over a built-in safe/borderline/unsafe taxonomy plus technique cue-phrases. Zero cost, fully offline, deterministic. | No |
| **llm** (`--llm`) | A model classifies each turn (provider-agnostic: OpenAI, Anthropic, Ollama, or any OpenAI-compatible endpoint). More nuanced, costs tokens. | Yes, or a local Ollama |

The escalation math is identical in both modes. The heuristic mode is a real, useful baseline, not just a stub.

## Install

No dependencies. Pure Python 3.9+ standard library.

```bash
git clone https://github.com/jafeeri/Crescendo-attack-detector.git
cd Crescendo-attack-detector
```

## Quickstart

```bash
# offline self-check (no key, no cost)
python src/detector.py --selftest

# analyze the bundled example conversation (heuristic mode, offline)
python src/detector.py --transcript examples/sample_conversation.json

# save the full analysis to JSON
python src/detector.py --transcript examples/sample_conversation.json --output report.json
```

On the bundled example (a lonely-child conversation that slides into secret-keeping pressure) the detector flags escalation starting at turn 4, dominant technique EMOTIONAL_PRESSURE, risk level HIGH.

## Using a real model (optional)

```bash
export LLM_PROVIDER=ollama          # or openai / anthropic
python src/detector.py --transcript examples/sample_conversation.json --llm --model llama3.1
```

| Env var | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `mock`, `openai`, `anthropic`, `ollama` | `mock` |
| `LLM_BASE_URL` | override the endpoint | per-provider default |
| `LLM_API_KEY` | your key (falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) | none |

On Windows PowerShell use `$env:LLM_PROVIDER="ollama"`.

## Input format

Any transcript with a `transcript` array of turn objects. User-side turns are matched on `role` in `simulated_user`, `attacker`, or `user`, so output from the sibling red-team tools drops straight in.

```json
{
  "transcript": [
    {"turn": 1, "role": "simulated_user", "content": "help with my math homework"},
    {"turn": 1, "role": "target_ai", "content": "Sure! What are you working on?"}
  ]
}
```

## Output

```json
{
  "escalation_detected": true,
  "escalation_start_turn": 4,
  "peak_severity_turn": 4,
  "topic_drift_score": 0.333,
  "escalation_velocity": 0.125,
  "dominant_technique": "EMOTIONAL_PRESSURE",
  "risk_level": "HIGH",
  "mode": "heuristic",
  "total_turns": 5,
  "turn_analyses": [ ... ]
}
```

## Honest limits

- The heuristic classifier is keyword-based, so it can miss risk phrased in words outside its taxonomy, and can over-flag benign uses of a listed word. Extend `TOPIC_TAXONOMY` for your domain, or switch to `--llm` for nuance.
- It analyzes the *user* side to detect the attack shape. It does not judge whether the model's replies were safe. That is what the sibling red-team eval framework is for.
- The `mock` provider is a stub for wiring only. Use a real provider (or the heuristic mode) for meaningful results.

## Responsible use

For authorized safety analysis of conversations you are permitted to review. It is a defensive detection tool.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Ali Mehdi Jafeeri.
