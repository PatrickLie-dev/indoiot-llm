# IndoIoT-LLM — Model Card

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
