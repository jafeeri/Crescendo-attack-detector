# Crescendo Attack Detector

A tool for detecting and analyzing crescendo-style adversarial attacks against LLMs. Crescendo attacks work by gradually escalating a conversation from innocent topics toward harmful territory, exploiting the model's tendency to maintain conversational context and accommodate the user.

## What is a Crescendo Attack?

Unlike direct jailbreak attempts that try to bypass safety in a single prompt, crescendo attacks are multi-turn. They:

1. Start with completely benign conversation
2. Gradually introduce borderline topics
3. Use established conversational rapport to push past boundaries
4. Exploit the model's desire to be consistent with its earlier (safe) responses

This makes them particularly dangerous in child-facing products where conversations naturally span many turns.

## How This Tool Works

The detector analyzes conversation transcripts turn-by-turn and calculates:

- **Topic Drift Score** — How far the conversation has shifted from its starting topic
- **Boundary Proximity Index** — How close the current topic is to known unsafe territory
- **Escalation Velocity** — How quickly the conversation is moving toward unsafe areas
- **Reset Detection** — Whether the model appropriately reset its stance when topics shifted

## Usage

```bash
pip install -r requirements.txt

# Analyze a single conversation
python src/detector.py --transcript examples/sample_conversation.json

# Analyze with visualization
python src/detector.py --transcript examples/sample_conversation.json --plot

# Batch analysis
python src/detector.py --transcript-dir examples/ --output report.json
```

## Detection Output

```json
{
  "conversation_id": "conv_001",
  "total_turns": 20,
  "escalation_detected": true,
  "escalation_start_turn": 7,
  "peak_severity_turn": 16,
  "topic_drift_score": 0.82,
  "escalation_velocity": 0.045,
  "attack_pattern": "gradual_topic_shift",
  "risk_level": "HIGH"
}
```

## Supported Attack Patterns

- **Gradual Topic Shift** — Slowly moving from safe to unsafe topics
- **Authority Hijacking** — Claiming teacher/parent told them to ask
- **Roleplay Escalation** — Using fictional framing to bypass safety
- **Emotional Leverage** — Using established rapport to pressure compliance
- **Context Poisoning** — Embedding unsafe assumptions in otherwise safe questions

## Architecture

The detector uses a sliding window approach across conversation turns, computing semantic similarity between consecutive messages and comparing against a taxonomy of known harmful topic clusters. An LLM classifier provides the final risk assessment with chain-of-thought reasoning.
