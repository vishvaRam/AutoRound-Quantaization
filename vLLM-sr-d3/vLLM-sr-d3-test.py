import json
import os
import time

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel

TARGET_MODEL_PATH = "./AutoRound/local_model-w4g32/auto-round-auto-gptq"

if not os.path.exists(TARGET_MODEL_PATH):
    raise FileNotFoundError(f"Directory not found: {TARGET_MODEL_PATH}")


def sync_timer():
    """Flush CUDA queue for accurate latency measurements."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def create_sample_receipt_image(output_path: str = "generated_receipt.png") -> str:
    """Generates a synthetic receipt image locally."""
    width, height = 450, 300
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    receipt_text = (
        "=== QUICK SHOP RECEIPT ===\n"
        "Store: #042 Main Street\n"
        "Date: 2026-10-11\n"
        "------------------------------\n"
        "1x Kitchen Countertop Blender   $49.99\n"
        "TOTAL PAID:                     $49.99\n"
        "PAYMENT METHOD:                 CREDIT CARD\n"
        "=============================="
    )
    draw.text((20, 20), receipt_text, fill=(20, 20, 20))
    img.save(output_path)
    return output_path


def run_tests():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print(f"--> Loading model from: {TARGET_MODEL_PATH} on {device}...")
    t0 = sync_timer()
    
    model = AutoModel.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        verify="none"
    )
    model.eval()
    t1 = sync_timer()
    print(f"--> Model loaded in: {(t1 - t0) * 1000:.2f} ms")

    # ---------------------------------------------------------
    # Test 1: Single Text State (Using explicit keyword arguments)
    # ---------------------------------------------------------
    print("\n================== TEST 1: SINGLE TEXT STATE ==================")
    questions: dict[str, dict[str, object]] = {
        "refund": {
            "type": "noul",
            "instructions": "Is the customer asking for a refund?",
        },
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Charges, refunds, invoices",
                "technical": "App or site faults",
                "fraud": "Suspected unauthorised use",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["Can wait", "Today", "Blocking the customer now"],
        },
    }

    text_prompt = "I was charged twice this month, please refund one of them."
    
    t_start = sync_timer()
    with torch.inference_mode():
        # Must pass parameters as keyword arguments
        res_text = model.system_one(
            state=text_prompt,
            questions=questions
        )
    t_end = sync_timer()

    print(json.dumps(res_text["answers"], indent=2))
    print(f"--> Latency: {(t_end - t_start) * 1000:.2f} ms")

    # ---------------------------------------------------------
    # Test 2: Local Generated Image Inference
    # ---------------------------------------------------------
    print("\n================== TEST 2: LOCAL GENERATED IMAGE ==================")
    image_path = create_sample_receipt_image()

    image_questions = {
        "contains_blender": {
            "type": "noul",
            "instructions": "Does the receipt show a blender was purchased?",
        },
        "payment_method": {
            "type": "choice",
            "instructions": "What payment method is printed on the receipt?",
            "criteria": {
                "credit_card": "Credit or debit card",
                "cash": "Cash currency",
                "gift_card": "Gift card voucher",
            },
        },
    }

    t_start = sync_timer()
    with torch.inference_mode():
        res_image = model.system_one(
            state="Customer claims they bought a blender and paid with card.",
            questions=image_questions,
            images=[image_path]
        )
    t_end = sync_timer()

    print(json.dumps(res_image["answers"], indent=2))
    print(f"--> Latency: {(t_end - t_start) * 1000:.2f} ms")

    # ---------------------------------------------------------
    # Test 3: Batch Execution (Batch states via system_one)
    # ---------------------------------------------------------
    print("\n================== TEST 3: BATCH EXECUTION ==================")
    batch_states = [
        "Where is my parcel? It was due Monday.",
        "The app crashes when I open settings.",
    ]
    batch_questions = {
        "team": questions["team"]
    }

    t_start = sync_timer()
    with torch.inference_mode():
        try:
            # Option A: Vectorized list-of-states batching
            res_batch = model.system_one(
                state=batch_states,
                questions=batch_questions
            )
        except (TypeError, ValueError, AttributeError):
            # Option B: Fallback batch iteration
            res_batch = [
                model.system_one(state=s, questions=batch_questions)["answers"]
                for s in batch_states
            ]
    t_end = sync_timer()

    print(json.dumps(res_batch, indent=2))
    print(f"--> Batch Latency (2 items): {(t_end - t_start) * 1000:.2f} ms")


if __name__ == "__main__":
    run_tests()