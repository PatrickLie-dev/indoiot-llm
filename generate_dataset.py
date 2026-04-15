"""
generate_dataset.py — IndoIoT LLM dataset generator using Groq API.

Usage:
    python generate_dataset.py          # full run (~650 samples)
    python generate_dataset.py preview  # preview existing dataset stats
    python generate_dataset.py test     # generate 5 samples only
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
OUTPUT_FILE = Path("dataset/indoiot_dataset.jsonl")
RATE_LIMIT_SLEEP = 2.5  # seconds between API calls (Groq free: 30 req/min)

SYSTEM_PROMPT = (
    "Kamu adalah seorang IoT Engineer senior yang ahli dalam hardware dan software IoT. "
    "Kamu memiliki pengalaman lebih dari 10 tahun dalam mengembangkan sistem IoT menggunakan "
    "ESP32, Arduino, MQTT, berbagai sensor dan aktuator, serta protokol komunikasi IoT. "
    "Jawab semua pertanyaan dalam Bahasa Indonesia yang jelas, teknis, dan mudah dipahami. "
    "Berikan contoh kode jika relevan."
)

# category → (target_samples, topic_seeds)
CATEGORIES: dict[str, tuple[int, list[str]]] = {
    "esp32_arduino": (
        150,
        [
            "GPIO dan pin configuration ESP32",
            "WiFi dan koneksi jaringan dengan ESP32",
            "Bluetooth BLE pada ESP32",
            "Deep sleep dan power management ESP32",
            "Timer dan interrupt pada Arduino/ESP32",
            "Pemrograman FreeRTOS pada ESP32",
            "OTA (Over-the-Air) update ESP32",
            "ADC dan DAC pada ESP32",
            "SPI dan I2C communication ESP32",
            "ESP32-CAM untuk computer vision IoT",
        ],
    ),
    "mqtt": (
        150,
        [
            "Konsep dasar MQTT broker dan client",
            "Quality of Service (QoS) levels di MQTT",
            "MQTT topic design dan best practices",
            "Keamanan dan autentikasi MQTT",
            "MQTT retained messages dan last will",
            "Implementasi MQTT dengan Mosquitto broker",
            "MQTT over WebSocket untuk web dashboard",
            "MQTT dengan TLS/SSL encryption",
            "Integrasi MQTT dengan Node-RED",
            "MQTT bridging dan clustering",
        ],
    ),
    "sensor_aktuator": (
        150,
        [
            "Sensor suhu dan kelembaban DHT22/BME280",
            "Sensor jarak ultrasonik HC-SR04",
            "Sensor PIR untuk deteksi gerakan",
            "Relay module untuk kontrol perangkat AC",
            "Servo motor dan stepper motor control",
            "Sensor gas MQ-series untuk kualitas udara",
            "Load cell dan HX711 untuk pengukuran berat",
            "Sensor cahaya LDR dan BH1750",
            "Sensor arus dan tegangan INA219/PZEM",
            "RFID RC522 untuk sistem akses kontrol",
        ],
    ),
    "protokol_iot": (
        100,
        [
            "Perbandingan MQTT vs HTTP vs CoAP untuk IoT",
            "LoRa dan LoRaWAN untuk IoT jarak jauh",
            "Zigbee dan Z-Wave untuk smart home",
            "Matter protocol untuk interoperabilitas IoT",
            "Modbus RTU/TCP untuk industrial IoT",
            "WebSocket untuk komunikasi real-time IoT",
            "gRPC vs REST untuk IoT microservices",
            "OPC-UA untuk industrial automation",
        ],
    ),
    "troubleshooting": (
        100,
        [
            "Debugging koneksi WiFi ESP32 yang tidak stabil",
            "Mengatasi memory leak pada Arduino/ESP32",
            "Troubleshooting MQTT connection drop",
            "Masalah brownout reset dan power supply ESP32",
            "Debug komunikasi I2C yang gagal",
            "Mengatasi watchdog timer reset pada ESP32",
            "Troubleshooting sensor pembacaan tidak akurat",
            "Masalah interrupt conflict pada mikrokontroler",
        ],
    ),
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def load_existing_samples(output_file: Path) -> dict[tuple[str, str], int]:
    """Return a counter of {(category, topic_seed): count} from disk."""
    existing: dict[tuple[str, str], int] = {}
    if not output_file.exists():
        return existing
    skipped = 0
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = (record["category"], record["topic_seed"])
                existing[key] = existing.get(key, 0) + 1
            except Exception:
                skipped += 1
                continue
    if skipped:
        logger.warning("Skipped %d malformed/unparseable lines in %s", skipped, output_file)
    total = sum(existing.values())
    logger.info("Loaded %d existing samples (%d unique seeds) from %s", total, len(existing), output_file)
    return existing


def generate_question(client: Groq, category: str, topic_seed: str) -> Optional[str]:
    """Ask Groq to generate one IoT question for the given topic."""
    user_prompt = (
        f"Buatkan SATU pertanyaan teknis tentang IoT dengan topik: '{topic_seed}'. "
        f"Pertanyaan harus spesifik, praktis, dan sesuai untuk level intermediate. "
        f"Tulis hanya pertanyaannya saja tanpa penjelasan tambahan."
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Failed to generate question for '%s': %s", topic_seed, exc)
        return None


def generate_answer(client: Groq, question: str) -> Optional[str]:
    """Ask Groq to generate a detailed answer for the given question."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Failed to generate answer: %s", exc)
        return None


def write_sample(
    f,
    instruction: str,
    response: str,
    category: str,
    topic_seed: str,
) -> None:
    """Write one JSONL record and flush immediately (resume-safe)."""
    record = {
        "instruction": instruction,
        "response": response,
        "category": category,
        "topic_seed": topic_seed,
    }
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def run_generation(
    client: Groq,
    output_file: Path,
    test_mode: bool = False,
) -> None:
    """Main generation loop. Supports full run and test mode (5 samples)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_samples(output_file)
    total_target = sum(n for n, _ in CATEGORIES.values())
    test_limit = 5
    generated = 0

    logger.info(
        "Starting generation | mode=%s | target=%d | already_done=%d",
        "TEST (5 samples)" if test_mode else "FULL",
        test_limit if test_mode else total_target,
        sum(existing.values()),
    )

    with output_file.open("a", encoding="utf-8") as f:
        for category, (target, seeds) in CATEGORIES.items():
            if test_mode and generated >= test_limit:
                break

            # How many samples per seed (distribute evenly, round up)
            samples_per_seed = -(-target // len(seeds))  # ceiling division

            for seed in seeds:
                if test_mode and generated >= test_limit:
                    break

                # How many samples to generate for this seed
                remaining_for_seed = samples_per_seed
                seed_generated = 0

                # Count already-generated for this (category, seed) pair
                already = existing.get((category, seed), 0)
                remaining_for_seed -= already

                if remaining_for_seed <= 0:
                    logger.info(
                        "Skipping '%s / %s' — already complete (%d samples)",
                        category,
                        seed,
                        already,
                    )
                    continue

                logger.info(
                    "Generating %d samples for [%s] '%s'",
                    remaining_for_seed,
                    category,
                    seed,
                )

                while seed_generated < remaining_for_seed:
                    if test_mode and generated >= test_limit:
                        break

                    # --- generate question ---
                    question = generate_question(client, category, seed)
                    if question is None:
                        logger.warning("Skipping sample due to question generation failure")
                        time.sleep(RATE_LIMIT_SLEEP)
                        continue
                    time.sleep(RATE_LIMIT_SLEEP)

                    # --- generate answer ---
                    answer = generate_answer(client, question)
                    if answer is None:
                        logger.warning("Skipping sample due to answer generation failure")
                        time.sleep(RATE_LIMIT_SLEEP)
                        continue
                    time.sleep(RATE_LIMIT_SLEEP)

                    write_sample(f, question, answer, category, seed)
                    seed_generated += 1
                    generated += 1
                    logger.info(
                        "  [%d] Saved sample | category=%s | Q: %s",
                        generated,
                        category,
                        question[:80] + ("..." if len(question) > 80 else ""),
                    )

    logger.info("Generation complete. Total new samples written: %d", generated)
    logger.info("Output: %s", output_file.resolve())


# ---------------------------------------------------------------------------
# Preview command
# ---------------------------------------------------------------------------


def run_preview(output_file: Path) -> None:
    """Print stats and a few samples from an existing dataset file."""
    if not output_file.exists():
        logger.error("Dataset file not found: %s", output_file)
        sys.exit(1)

    records: list[dict] = []
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Stats
    category_counts: dict[str, int] = {}
    for r in records:
        cat = r.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Dataset: {output_file}")
    print(f"  Total samples: {len(records)}")
    print(f"{'='*60}")
    for cat, count in sorted(category_counts.items()):
        target = CATEGORIES.get(cat, (0, []))[0]
        pct = count / target * 100 if target else 0
        bar = "#" * int(pct / 5)
        print(f"  {cat:<22} {count:>4}/{target:<4} [{bar:<20}] {pct:5.1f}%")
    print(f"{'='*60}\n")

    # Show first 3 samples
    print("Sample records (first 3):\n")
    for i, r in enumerate(records[:3], 1):
        print(f"--- Sample {i} [{r.get('category')}] ---")
        print(f"Q: {r.get('instruction', '')[:120]}")
        print(f"A: {r.get('response', '')[:200]}...")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "full"

    if mode == "preview":
        run_preview(OUTPUT_FILE)
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error(
            "GROQ_API_KEY not set. Copy .env.example → .env and add your key."
        )
        sys.exit(1)

    client = Groq(api_key=api_key)

    if mode == "test":
        logger.info("TEST MODE — generating 5 samples only")
        run_generation(client, OUTPUT_FILE, test_mode=True)
    elif mode == "full":
        logger.info("FULL MODE — generating complete dataset (~650 samples)")
        logger.info("Estimated time: ~55 minutes (rate-limited to 30 req/min)")
        run_generation(client, OUTPUT_FILE, test_mode=False)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
