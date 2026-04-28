---
title: IndoIoT LLM
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
python_version: "3.11"
---

# IndoIoT LLM — Asisten IoT Berbahasa Indonesia

> Fine-tuned **Qwen2.5-3B-Instruct** dengan QLoRA pada 760 samples IoT Bahasa Indonesia. Proyek portfolio AI Engineer end-to-end: dari pembuatan dataset, fine-tuning, experiment tracking, hingga deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-yellow)
![Gradio](https://img.shields.io/badge/Gradio-5.x-purple)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow)

---

## Daftar Isi

- [Overview](#overview)
- [Demo](#demo)
- [Fitur Utama](#fitur-utama)
- [Tech Stack](#tech-stack)
- [Arsitektur](#arsitektur)
- [Dataset](#dataset)
- [Konfigurasi Training](#konfigurasi-training)
- [Cara Menjalankan](#cara-menjalankan)
- [Screenshots](#screenshots)
- [Hasil Perbandingan Model](#hasil-perbandingan-model)
- [Struktur Proyek](#struktur-proyek)
- [Kontak](#kontak)

---

## Overview

**IndoIoT LLM** adalah model bahasa besar (LLM) berbahasa Indonesia yang dikhususkan untuk domain **IoT (Internet of Things)**. Model ini dibangun dengan melakukan fine-tuning pada **Qwen2.5-3B-Instruct** menggunakan teknik **QLoRA (4-bit NF4 + LoRA)**, memungkinkan training efisien pada consumer GPU (RTX 4060 Laptop, 8GB VRAM).

Proyek ini mencakup seluruh pipeline MLOps secara end-to-end:

```
Dataset Generation → Fine-Tuning → Experiment Tracking → Deployment
     (Groq API)       (QLoRA/SFT)      (MLflow)          (Gradio + Docker + HF Hub)
```

Model mampu menjawab pertanyaan teknis IoT seperti:
- Koneksi ESP32 ke broker MQTT dengan library PubSubClient
- Perbedaan QoS 0, 1, dan 2 dalam protokol MQTT
- Cara membaca sensor DHT22 dan mengirim data ke server
- Troubleshooting watchdog timer reset pada ESP32
- Perbandingan protokol HTTP vs MQTT untuk IoT

---

## Demo

Model tersedia di HuggingFace Hub: **[Pat-L/indoiot-qwen2.5-lora](https://huggingface.co/Pat-L/indoiot-qwen2.5-lora)**

Live demo (Gradio Space): **[Pat-L/indoiot-llm](https://huggingface.co/spaces/Pat-L/indoiot-llm)**

---

## Fitur Utama

- **Bahasa Indonesia native** — dilatih khusus untuk menjawab pertanyaan IoT dalam Bahasa Indonesia yang teknis dan mudah dipahami
- **QLoRA fine-tuning** — 4-bit quantization (NF4) + LoRA dengan hanya ~1% parameter yang dilatih, efisien untuk consumer GPU
- **Dataset buatan sendiri** — 760 sampel IoT dihasilkan menggunakan Groq API (LLaMA 3.1) dengan 5 kategori topik
- **Experiment tracking** — integrasi MLflow untuk melacak hyperparameter, loss, dan konfigurasi training
- **Gradio web UI** — antarmuka chat yang siap digunakan, mendukung GPU dan CPU
- **Docker ready** — deployment mudah dengan container pre-configured
- **Side-by-side comparison** — skrip inference untuk membandingkan respons base model vs fine-tuned model

---

## Tech Stack

| Kategori | Library / Tool | Keterangan |
|---|---|---|
| **Base Model** | Qwen2.5-3B-Instruct | 3B parameter, ChatML format |
| **Fine-Tuning** | PEFT + TRL (SFTTrainer) | LoRA adapter training |
| **Quantization** | BitsAndBytes | 4-bit NF4 quantization |
| **Framework** | PyTorch 2.6 + CUDA 12.4 | GPU acceleration |
| **Tokenizer/Model** | Transformers 5.x | HuggingFace Transformers |
| **Dataset** | Groq API (LLaMA 3.1 8B) | Synthetic dataset generation |
| **Experiment Tracking** | MLflow | Training metrics & artifacts |
| **Web UI** | Gradio 5.x | Interactive chat interface |
| **Model Registry** | HuggingFace Hub | Model hosting & versioning |
| **Containerization** | Docker | Reproducible deployment |
| **Language** | Python 3.11 | |

---

## Arsitektur

```
┌─────────────────────────────────────────────────────────────────┐
│                       PIPELINE END-TO-END                       │
└─────────────────────────────────────────────────────────────────┘

  1. DATASET GENERATION
     ┌────────────────────────────────────────────────┐
     │  generate_dataset.py                           │
     │                                                │
     │  Groq API (LLaMA 3.1 8B)                      │
     │    ├── 5 kategori topik IoT                   │
     │    ├── ~650-760 pasangan Q&A                  │
     │    └── Format JSONL (resume-safe)             │
     └─────────────────────┬──────────────────────────┘
                           │  dataset/indoiot_dataset.jsonl
                           ▼
  2. FINE-TUNING
     ┌────────────────────────────────────────────────┐
     │  train.py                                      │
     │                                                │
     │  Qwen2.5-3B-Instruct (base)                   │
     │    │                                           │
     │    ├── 4-bit NF4 Quantization (BnB)           │
     │    ├── LoRA: r=16, alpha=32                   │
     │    │   target: q/k/v/o_proj                   │
     │    └── SFTTrainer (3 epochs, lr=2e-4)         │
     │                                                │
     │  Output: models/lora-adapter-only/            │
     └─────────────────────┬──────────────────────────┘
                           │  LoRA adapter weights
                           ▼
  3. EXPERIMENT TRACKING
     ┌────────────────────────────────────────────────┐
     │  mlflow_portfolio/                             │
     │    ├── Hyperparameter logging                 │
     │    ├── Training config artifacts              │
     │    └── Run comparison dashboard              │
     └─────────────────────┬──────────────────────────┘
                           │
                           ▼
  4. DEPLOYMENT
     ┌────────────────────────────────────────────────┐
     │  app.py + Dockerfile                           │
     │                                                │
     │  HuggingFace Hub (Pat-L/indoiot-qwen2.5-lora) │
     │    │                                           │
     │    ├── Gradio Web UI (port 7860)              │
     │    ├── GPU: 4-bit inference                   │
     │    └── CPU: float16 fallback                 │
     └────────────────────────────────────────────────┘
```

### Alur Inference (Runtime)

```
User Input (Bahasa Indonesia)
        │
        ▼
  ChatML Prompt Format
  <|im_start|>system
  Kamu adalah asisten IoT expert Indonesia...
  <|im_end|>
  <|im_start|>user
  {pertanyaan}
  <|im_end|>
  <|im_start|>assistant
        │
        ▼
  Qwen2.5-3B + LoRA Adapter
  (temperature=0.7, max_new_tokens=300)
        │
        ▼
  Respons teknis IoT dalam Bahasa Indonesia
```

---

## Dataset

Dataset dihasilkan secara otomatis menggunakan Groq API dengan model LLaMA 3.1 8B Instant.

| Kategori | Jumlah Target | Topik Utama |
|---|---|---|
| `esp32_arduino` | 150 | GPIO, WiFi, BLE, FreeRTOS, OTA, ADC/DAC, SPI/I2C |
| `mqtt` | 150 | QoS levels, broker, keamanan, TLS/SSL, Node-RED |
| `sensor_aktuator` | 150 | DHT22/BME280, HC-SR04, PIR, relay, servo, RFID |
| `protokol_iot` | 100 | MQTT vs HTTP vs CoAP, LoRaWAN, Zigbee, Matter, OPC-UA |
| `troubleshooting` | 100 | WiFi instability, memory leak, watchdog, I2C debug |
| **Total** | **~760** | |

Format dataset:
```json
{
  "instruction": "Bagaimana cara menghubungkan ESP32 ke broker MQTT?",
  "response": "Untuk menghubungkan ESP32 ke broker MQTT...",
  "category": "mqtt",
  "topic_seed": "Konsep dasar MQTT broker dan client"
}
```

---

## Konfigurasi Training

**LoRA Config:**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Quantization (BitsAndBytes):**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

**SFT Training:**
```
Epochs:               3
Batch size:           2 (gradient accumulation: 4 → effective batch: 8)
Learning rate:        2e-4
LR Scheduler:         cosine
Max sequence length:  512
Trainable params:     ~1% (LoRA only)
Training GPU:         RTX 4060 Laptop (8GB VRAM)
Estimated duration:   ~45-55 menit
```

---

## Cara Menjalankan

### Prerequisites

```bash
git clone https://github.com/pukiqq/indoiot-llm.git
cd indoiot-llm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Buat file `.env`:
```bash
cp .env.example .env
# Isi HF_TOKEN dan GROQ_API_KEY di .env
```

---

### 1. Generate Dataset (Opsional — dataset sudah tersedia)

```bash
pip install groq python-dotenv

# Preview dataset yang ada
python generate_dataset.py preview

# Test generate 5 samples
python generate_dataset.py test

# Generate dataset lengkap (~55 menit)
python generate_dataset.py
```

---

### 2. Fine-Tuning

> Membutuhkan GPU NVIDIA dengan CUDA 12.x dan minimal 8GB VRAM.

```bash
pip install -r requirements.txt
pip install bitsandbytes trl datasets

python train.py
```

Model akan tersimpan di `./models/lora-adapter-only/`.

---

### 3. Inference — Perbandingan Base vs Fine-Tuned

```bash
# Side-by-side comparison (base model vs fine-tuned)
python inference.py

# Fine-tuned saja (lebih cepat)
python inference.py --finetuned-only
```

Hasil tersimpan di `results/comparison.txt`.

---

### 4. Jalankan Web UI (Gradio)

**Cara A — Langsung:**
```bash
pip install -r requirements.txt

# Tambahkan HF_TOKEN di .env (untuk load model dari HuggingFace Hub)
python app.py
```

Buka browser di `http://localhost:7860`

**Cara B — Docker (CPU):**
```bash
docker build -t indoiot-llm .
docker run -p 7860:7860 -e HF_TOKEN=your_token indoiot-llm
```

---

### 5. MLflow Experiment Tracking

```bash
pip install mlflow

# Jalankan MLflow UI
mlflow ui --backend-store-uri ./mlflow_portfolio/mlruns

# Buka browser di http://localhost:5000
```

---

## Screenshots

> *(Tambahkan screenshot setelah deployment)*

**Gradio Chat Interface:**

```
[ Screenshot: tampilan chat UI dengan contoh pertanyaan IoT ]
```

**MLflow Dashboard:**

```
[ Screenshot: experiment tracking dengan training metrics ]
```

**Inference Comparison:**

```
[ Screenshot: perbandingan respons base model vs fine-tuned ]
```

---

## Hasil Perbandingan Model

Contoh perbandingan respons untuk pertanyaan IoT:

**Q: Apa perbedaan antara QoS 0, 1, dan 2 dalam protokol MQTT?**

| | Base Model (Qwen2.5-3B) | Fine-Tuned (IndoIoT LLM) |
|---|---|---|
| **Gaya** | Formal, general | Teknis, domain-specific |
| **Struktur** | Paragraf panjang | Heading jelas per level QoS |
| **Kode** | Tidak ada | Ada contoh Python (paho-mqtt) |
| **Bahasa** | Bahasa Indonesia campur | Bahasa Indonesia konsisten |

Lihat hasil lengkap di [`results/comparison.txt`](results/comparison.txt).

---

## Struktur Proyek

```
indoiot-llm/
├── app.py                  # Gradio web UI (load dari HuggingFace Hub)
├── train.py                # Fine-tuning dengan QLoRA + SFTTrainer
├── inference.py            # Side-by-side comparison base vs fine-tuned
├── generate_dataset.py     # Dataset generation via Groq API
├── save_checkpoint.py      # Simpan checkpoint tertentu
├── push_to_hub.py          # Upload model ke HuggingFace Hub
├── requirements.txt        # Dependencies inference + UI
├── Dockerfile              # Container deployment (CPU)
├── .env.example            # Template environment variables
├── dataset/
│   └── indoiot_dataset.jsonl  # Training dataset (~760 samples)
├── models/
│   ├── indoiot-qwen2.5-3b-lora/   # Full fine-tuned model
│   └── lora-adapter-only/          # LoRA adapter weights saja
├── results/
│   └── comparison.txt      # Output inference comparison
└── mlflow_portfolio/
    └── mlruns/             # MLflow experiment tracking data
```

---

## Kontak

**Patrick** — AI Engineer Portfolio  
GitHub: [@pukiqq](https://github.com/pukiqq)  
HuggingFace: [Pat-L](https://huggingface.co/Pat-L)  
Email: patricklie995@gmail.com

---

*Proyek ini dibuat sebagai bagian dari portfolio AI Engineer, mendemonstrasikan kemampuan end-to-end MLOps: dataset engineering, LLM fine-tuning, experiment tracking, dan production deployment.*
