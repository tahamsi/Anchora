# Anchora

UGRR (Uncertainty-Gated RL with Regret Replay) pipeline for fine-tuning instruction-following LLMs.

## Problem and Goal
Modern LLMs can be overconfident on hard prompts and drift during RL fine-tuning. UGRR tackles this by:
- Scoring uncertainty on model outputs.
- Detecting regret cases where the model likely failed.
- Replaying those cases with KL-regularized updates to improve reliability without catastrophic drift.

The goal is a practical, single-GPU pipeline you can run end-to-end: SFT -> reward models -> uncertainty/regret -> RL updates -> evaluation.

## Solution Overview
The pipeline is modular and runs in stages:
- **SFT**: fine-tune a base LLM on instruction data.
- **Reward/verifier models**: train preference-based scorers.
- **Uncertainty scoring**: entropy + MC dropout diversity + verifier disagreement.
- **Regret replay buffer**: collect uncertain or low-score outputs for focused updates.
- **KL-regularized DPO updates**: improve policy while staying close to a reference.
- **Evaluation**: proxy scoring on AlpacaEval/MT-Bench and exact match on GSM8K.

## Models Used
Defaults are configurable via `config.json`/`config.yaml`:
- **Policy / SFT base**: `mistral-7b` (any causal LM compatible with `AutoModelForCausalLM`)
- **Reward model**: `bert-base-uncased` (sequence classifier; you can swap for a different scorer)
- **Verifier ensemble**: optional list of trained reward models

## Setup
Install dependencies:

```bash
pip install torch transformers datasets pytest pyyaml
```

## Configuration
Create a JSON or YAML config file with overrides. Example `config.json`:

```json
{
  "base_model_name": "mistral-7b",
  "reward_model_name": "bert-base-uncased",
  "sft_data_path": "data/sft.json",
  "preference_data_path": "data/preferences.json",
  "output_dir": "models",
  "log_dir": "logs",
  "batch_size": 2,
  "sft_epochs": 1,
  "reward_epochs": 1,
  "rl_iterations": 100,
  "max_length": 512
}
```

## Data Formats
SFT data expects JSON/CSV with `prompt` and `completion` (or `instruction` and `output`).

Preference data expects JSON/CSV with:
- `prompt`
- `response_a`
- `response_b`
- `label` (1 if A preferred, 0 if B preferred)

## Commands
SFT:

```bash
python ugrr_pipeline.py sft --config config.json
```

Reward model training:

```bash
python ugrr_pipeline.py train_reward --config config.json
```

UGRR regret loop (DPO + KL):

```bash
python ugrr_pipeline.py regret_loop --config config.json
```

Evaluation:

```bash
python ugrr_pipeline.py evaluate --config config.json --model_path models/ugrr_final --baseline_path models/mistral_sft
```

Unit tests:

```bash
pytest -q
```

## Training Considerations on a Large-Memory GPU
To run comfortably on a high-memory GPU without naming hardware:
- **Precision**: Enable `use_bf16` in config to reduce memory use.
- **Batch size**: Increase `batch_size` cautiously; prefer higher `grad_accum_steps` over very large batch sizes.
- **Sequence length**: Keep `max_length` tight (e.g., 512 or 1024) to control memory and speed.
- **Gradient checkpointing**: Keep `gradient_checkpointing=True` for larger models.
- **Dropout sampling**: Uncertainty uses MC dropout; keep `uncertainty_dropout_samples` modest (e.g., 4-8).
- **Eval limits**: Use small `limit` values in evaluation when iterating quickly.
- **KL control**: Tune `kl_beta_min`/`kl_beta_max` to prevent drift; larger values are safer when uncertainty is high.

## Running the Full Application
Minimal full run, assuming data exists:

```bash
python ugrr_pipeline.py sft --config config.json
python ugrr_pipeline.py train_reward --config config.json
python ugrr_pipeline.py regret_loop --config config.json
python ugrr_pipeline.py evaluate --config config.json --model_path models/ugrr_final --baseline_path models/mistral_sft
```

Artifacts and logs:
- Models are written under `models/`
- Logs and configs are written under `logs/`

## Notes
- AlpacaEval/MT-Bench use reward model proxies unless you integrate a judge API.
- GSM8K evaluation extracts the last numeric answer from the model output.
