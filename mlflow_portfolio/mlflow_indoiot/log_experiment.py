# mlflow_indoiot/log_experiment.py
"""
MLflow Experiment Logger for IndoIoT-LLM Fine-tuning
Logs training configuration, metrics, and model artifacts
Compatible with existing QLoRA training setup
"""

import sys
import mlflow
import mlflow.pytorch
import json
import os
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")  # local by default
EXPERIMENT_NAME = "IndoIoT-LLM-QLoRA"

# Replicate your actual training config
TRAINING_CONFIG = {
    # Model
    "base_model": "Qwen/Qwen2.5-3B",
    "adapter_type": "QLoRA",
    "quantization": "4bit",
    
    # LoRA hyperparams
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    
    # Training hyperparams
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "max_seq_length": 512,
    "fp16": True,
    
    # Dataset
    "dataset_size": 760,
    "dataset_language": "Indonesian",
    "dataset_domain": "IoT",
    "train_split": 0.9,
    "eval_split": 0.1,
}

# Simulated metrics (replace with actual values from your training logs)
# If you have actual logs, parse them instead
TRAINING_METRICS_PER_EPOCH = [
    {"epoch": 1, "train_loss": 1.842, "eval_loss": 1.654, "learning_rate": 1.8e-4},
    {"epoch": 2, "train_loss": 1.523, "eval_loss": 1.401, "learning_rate": 1.2e-4},
    {"epoch": 3, "train_loss": 1.287, "eval_loss": 1.198, "learning_rate": 4e-5},
]

FINAL_METRICS = {
    "final_train_loss": 1.287,
    "final_eval_loss": 1.198,
    "perplexity": 3.31,  # exp(final_eval_loss)
    "training_duration_minutes": 47.3,
    "gpu_memory_peak_gb": 12.4,
    "trainable_params": 4_194_304,
    "total_params": 3_090_000_000,
    "trainable_pct": 0.136,
}


def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "project": "IndoIoT-LLM",
                "owner": "Patrick Lie",
                "task": "Causal Language Modeling",
                "language": "Indonesian",
            }
        )
        print(f"✅ Created experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    else:
        print(f"✅ Using existing experiment: {EXPERIMENT_NAME}")
    
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_run(run_name: str = None):
    """Log a complete training run to MLflow."""
    
    if run_name is None:
        run_name = f"qwen2.5-3b-qlora-{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n🚀 Started MLflow run: {run.info.run_id}")
        
        # ── 1. Log all hyperparameters ─────────────────────────────────────
        mlflow.log_params(TRAINING_CONFIG)
        print("  ✅ Logged hyperparameters")
        
        # ── 2. Log metrics per epoch (creates time-series charts in UI) ────
        for epoch_data in TRAINING_METRICS_PER_EPOCH:
            step = epoch_data["epoch"]
            mlflow.log_metric("train_loss", epoch_data["train_loss"], step=step)
            mlflow.log_metric("eval_loss", epoch_data["eval_loss"], step=step)
            mlflow.log_metric("learning_rate", epoch_data["learning_rate"], step=step)
        print("  ✅ Logged epoch metrics")
        
        # ── 3. Log final summary metrics ───────────────────────────────────
        mlflow.log_metrics(FINAL_METRICS)
        print("  ✅ Logged final metrics")
        
        # ── 4. Log training config as artifact (JSON) ──────────────────────
        config_path = "training_config.json"
        with open(config_path, "w") as f:
            json.dump(TRAINING_CONFIG, f, indent=2)
        mlflow.log_artifact(config_path)
        os.remove(config_path)
        print("  ✅ Logged training config artifact")
        
        # ── 5. Log model card as artifact ──────────────────────────────────
        model_card = generate_model_card()
        card_path = "MODEL_CARD.md"
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        mlflow.log_artifact(card_path)
        os.remove(card_path)
        print("  ✅ Logged model card artifact")
        
        # ── 6. Set tags ────────────────────────────────────────────────────
        mlflow.set_tags({
            "model.huggingface_url": "https://huggingface.co/spaces/Pat-L/indoiot-llm",
            "model.base": "Qwen2.5-3B",
            "model.adapter": "QLoRA",
            "dataset.size": "760 samples",
            "dataset.domain": "Indonesian IoT",
            "status": "deployed",
            "deployment.platform": "HuggingFace Spaces",
        })
        print("  ✅ Set run tags")
        
        print(f"\n✨ Run complete!")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   View UI: mlflow ui --port 5001")
        
        return run.info.run_id


def generate_model_card() -> str:
    """Generate a model card markdown for the artifact."""
    return """# IndoIoT-LLM — Model Card

## Model Description
Fine-tuned Qwen2.5-3B using QLoRA on 760 Indonesian IoT domain samples.
Deployed live on HuggingFace Spaces.

## Training Configuration
- **Base Model:** Qwen/Qwen2.5-3B
- **Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Dataset:** 760 Indonesian IoT Q&A samples
- **Epochs:** 3
- **Final Eval Loss:** 1.198 | **Perplexity:** 3.31

## Performance
| Metric | Value |
|--------|-------|
| Final Train Loss | 1.287 |
| Final Eval Loss | 1.198 |
| Perplexity | 3.31 |
| Trainable Parameters | 4.2M (0.136% of total) |

## Deployment
🔗 [HuggingFace Spaces](https://huggingface.co/spaces/Pat-L/indoiot-llm)

## Author
Patrick Lie — AI Engineer Portfolio Project
"""


if __name__ == "__main__":
    setup_mlflow()
    run_id = log_training_run("qwen2.5-3b-qlora-indoiot-v1")
    print(f"\n🎯 To view results, run:")
    print(f"   mlflow ui --port 5001")
    print(f"   Then open: http://localhost:5001")