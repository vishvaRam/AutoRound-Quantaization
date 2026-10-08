import os

import torch
from transformers import AutoModel
from transformers.image_utils import load_image

# ---------------------------------------------------------
# Target quantized export directory to verify
# ---------------------------------------------------------
TARGET_MODEL_PATH = "./AutoRound/local_model-w4g32/auto-round-auto-gptq"
# Alternatives:
# TARGET_MODEL_PATH = "./AutoRound/local_model-w4g32/auto-gptq"
# TARGET_MODEL_PATH = "./AutoRound/local_model-w4g32/llm-compressor-wint-a16"

if not os.path.exists(TARGET_MODEL_PATH):
    raise FileNotFoundError(f"Directory not found: {TARGET_MODEL_PATH}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32 if device == "cpu" else torch.bfloat16

print(f"Loading quantized model from: {TARGET_MODEL_PATH} ...")
model = AutoModel.from_pretrained(
    TARGET_MODEL_PATH,
    trust_remote_code=True,
    dtype=dtype,
).to("cuda")

if device != "cuda":
    model = model.to(device)

# ---------------------------------------------------------
# Test Cases
# ---------------------------------------------------------

# 1. Text multi-question inference
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

print("\n--- Testing Single Text State ---")
text_prompt = "I was charged twice this month, please refund one of them."
res_text = model.system_one(text_prompt, questions)
print(res_text)

# 2. Vision inference
print("\n--- Testing Image Input ---")
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(image_url)

cats_question: dict[str, object] = {
    "type": "choice",
    "instructions": "How many cats are there?",
    "criteria": {"one": "One", "two": "Two", "more": "Three or more"},
}
res_image = model.system_one(None, {"cats": cats_question}, images=[image])
print(res_image)

# 3. Packed batch inference
print("\n--- Testing Batch Execution ---")
tickets: list[str] = [
    "Where is my parcel? It was due Monday.",
    "The app crashes when I open settings.",
]
batch_payload: list[tuple[str, dict[str, object]]] = [
    (ticket, {"team": questions["team"]}) for ticket in tickets
]
res_batch = model.system_one_batch(batch_payload)
print(res_batch)