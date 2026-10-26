#!/usr/bin/env python3
"""
Fast Direct Hugging Face Pipeline for Marathi NLP
-------------------------------------------------
Bypasses the heavy mahaNLP package wrappers by directly utilizing Hugging Face's
optimized pipelines and caching mechanisms.

Supported Tasks:
  - sentiment  : Marathi Sentiment Analysis (l3cube-pune/marathi-sentiment-md)
  - hate       : Hate & Offensive Speech Detection (l3cube-pune/mahahate-bert)
  - ner        : Named Entity Recognition (l3cube-pune/marathi-ner)
  - mask-fill  : Masked Word Prediction (l3cube-pune/marathi-bert)
"""

import sys
import time
import argparse

# Default supported models from L3Cube Hugging Face Hub
TASK_MODELS = {
    "sentiment": {
        "model_id": "l3cube-pune/marathi-sentiment-md",
        "pipeline_task": "sentiment-analysis",
        "samples": [
            ("दिवाळीच्या सणादरम्यान सगळे आनंदी असतात.", "Everyone is happy during Diwali festival."),
            ("मी तुझा तिरस्कार करतो.", "I hate you."),
            ("मी आज कॉलेजला जात आहे.", "I am going to college today."),
        ],
    },
    "hate": {
        "model_id": "l3cube-pune/mahahate-bert",
        "pipeline_task": "text-classification",
        "samples": [
            ("ती मूर्ख आहे. मला ती अजिबात आवडत नाही.", "She is stupid. I do not like her at all."),
            ("मला तुझे खूप कौतुक वाटते, खूप छान काम केलेस.", "I appreciate you a lot, you did great work."),
        ],
    },
    "ner": {
        "model_id": "l3cube-pune/marathi-ner",
        "pipeline_task": "token-classification",
        "samples": [
            ("भारताचे दुसरे राष्ट्रपती डॉ. सर्वपल्ली राधाकृष्णन यांचा जन्मदिवस शिक्षक दिन म्हणून साजरा करण्यात येतो.",
             "Second President of India Dr. Sarvepalli Radhakrishnan's birthday is celebrated as Teachers Day."),
            ("सचिन तेंडुलकर मुंबईत राहतात.", "Sachin Tendulkar lives in Mumbai."),
        ],
    },
    "mask-fill": {
        "model_id": "l3cube-pune/marathi-bert",
        "pipeline_task": "fill-mask",
        "samples": [
            ("मी महाराष्ट्रात [MASK].", "I [MASK] in Maharashtra."),
            ("भारताची राजधानी नवी [MASK] आहे.", "The capital of India is New [MASK]."),
        ],
    },
}


def load_pipeline(task_name: str):
    """Load the Hugging Face pipeline for the specified task."""
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        print("\n[ERROR] Required packages not found!")
        print("Please run: pip install torch transformers")
        sys.exit(1)

    task_config = TASK_MODELS[task_name]
    model_id = task_config["model_id"]
    pipeline_task = task_config["pipeline_task"]

    device = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA GPU" if device == 0 else "CPU"

    print(f"\n=======================================================")
    print(f"Loading task   : {task_name.upper()}")
    print(f"Model ID       : {model_id}")
    print(f"Running device : {device_name}")
    print(f"=======================================================")

    t0 = time.time()
    pipe = pipeline(
        task=pipeline_task,
        model=model_id,
        device=device,
        aggregation_strategy="simple" if pipeline_task == "token-classification" else None,
    )
    load_time = time.time() - t0
    print(f"[✓] Pipeline ready in {load_time:.2f} seconds.")
    return pipe, task_config


def run_samples(pipe, task_name: str, task_config: dict):
    """Run predefined Marathi test samples and measure inference speed."""
    print(f"\n--- Running Demo Samples for '{task_name}' ---")
    samples = task_config["samples"]

    for text, translation in samples:
        t_start = time.time()
        result = pipe(text)
        duration_ms = (time.time() - t_start) * 1000

        print(f"\nInput (Marathi)  : {text}")
        print(f"Translation      : {translation}")
        print(f"Inference Time   : {duration_ms:.1f} ms")

        if task_name == "ner":
            entities = [f"{item['word']} ({item['entity_group']}: {item['score']:.2f})" for item in result]
            print(f"Entities Detected: {', '.join(entities) if entities else 'None'}")
        elif task_name == "mask-fill":
            top_preds = [f"'{r['token_str'].strip()}' ({r['score']:.2%})" for r in result[:3]]
            print(f"Top Predictions  : {', '.join(top_preds)}")
        else:
            label = result[0].get("label", "Unknown")
            score = result[0].get("score", 0.0)
            print(f"Prediction       : {label} (Confidence: {score:.2%})")


def interactive_loop(pipe, task_name: str):
    """Allow custom user input for real-time testing."""
    print(f"\n--- Interactive Mode for '{task_name}' ---")
    print("Type your Marathi sentence and press Enter (or type 'quit' / 'exit' to stop):")

    while True:
        try:
            user_input = input("\nEnter Marathi text > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            print("Done.")
            break

        t_start = time.time()
        result = pipe(user_input)
        duration_ms = (time.time() - t_start) * 1000

        print(f"Time: {duration_ms:.1f} ms")
        if task_name == "ner":
            entities = [f"{item['word']} -> {item['entity_group']} ({item['score']:.2f})" for item in result]
            print(f"Entities: {', '.join(entities) if entities else 'None found'}")
        elif task_name == "mask-fill":
            for i, r in enumerate(result[:3], 1):
                print(f"  {i}. {r['sequence']} (score: {r['score']:.2%})")
        else:
            print(f"Result: {result[0]['label']} (Confidence: {result[0]['score']:.2%})")


def main():
    parser = argparse.ArgumentParser(
        description="Fast Direct Hugging Face Pipeline for Marathi NLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["sentiment", "hate", "ner", "mask-fill", "all"],
        default="sentiment",
        help="NLP Task to test (default: sentiment)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enter interactive prompt to test custom Marathi sentences",
    )

    args = parser.parse_args()

    tasks_to_run = list(TASK_MODELS.keys()) if args.task == "all" else [args.task]

    for task in tasks_to_run:
        pipe, config = load_pipeline(task)
        run_samples(pipe, task, config)
        if args.interactive and args.task != "all":
            interactive_loop(pipe, task)

    print("\n[✓] Completed successfully!")


if __name__ == "__main__":
    main()
