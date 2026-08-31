import json
import os
import shutil


from huggingface_hub import HfApi, create_repo, get_token, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
BASE_MODEL_ID = "Qwen/Qwen3.8-27B"                              # Unquantized base model
QUANT_MODEL_ID = "Vishva007/Qwen3.8-27B-W4A16-AutoRound"        # Quantized repo to fix
NEW_REPO_ID = "Vishva007/Qwen3.8-27B-W4A16-AutoRound"


LOCAL_BASE_DIR = "./base_model"
LOCAL_QUANT_DIR = "./quant_model"
OUTPUT_FIXED_DIR = "./fixed_model"

HF_TOKEN = os.getenv("HF_TOKEN")


# -------------------------------------------------------------
# 1. Download base & quantized models
# -------------------------------------------------------------
print("Downloading base model and quantized checkpoint...")
snapshot_download(
    repo_id=BASE_MODEL_ID, local_dir=LOCAL_BASE_DIR, token=HF_TOKEN
)
snapshot_download(
    repo_id=QUANT_MODEL_ID, local_dir=LOCAL_QUANT_DIR, token=HF_TOKEN
)


os.makedirs(OUTPUT_FIXED_DIR, exist_ok=True)


# Copy all non-safetensor metadata files
for item in os.listdir(LOCAL_QUANT_DIR):
  s = os.path.join(LOCAL_QUANT_DIR, item)
  d = os.path.join(OUTPUT_FIXED_DIR, item)
  if os.path.isfile(s) and not item.endswith(".safetensors"):
    shutil.copy2(s, d)


# -------------------------------------------------------------
# 2. Extract unquantized MTP tensors from base model
# -------------------------------------------------------------
print("\nExtracting native BF16 MTP weights from base model...")
base_mtp_tensors = {}


for file in os.listdir(LOCAL_BASE_DIR):
  if file.endswith(".safetensors"):
    file_path = os.path.join(LOCAL_BASE_DIR, file)
    with safe_open(file_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        if "mtp" in key.lower():
          base_mtp_tensors[key] = f.get_tensor(key)
          print(f"  Found MTP tensor: {key} ({base_mtp_tensors[key].dtype})")


# -------------------------------------------------------------
# 3. Graft tensors into the quantized checkpoint
# -------------------------------------------------------------
print("\nGrafting weights into quantized shards...")


quant_safetensors = [
    f for f in os.listdir(LOCAL_QUANT_DIR) if f.endswith(".safetensors")
]
is_single_shard = len(quant_safetensors) == 1


if is_single_shard:
  shard_name = quant_safetensors[0]
  quant_tensors = {}
  with safe_open(
      os.path.join(LOCAL_QUANT_DIR, shard_name), framework="pt", device="cpu"
  ) as f:
    for k in f.keys():
      if "mtp" not in k.lower():  # drop quantized 4-bit MTP tensors
        quant_tensors[k] = f.get_tensor(k)


  # Inject base BF16 tensors
  quant_tensors.update(base_mtp_tensors)
  save_file(
      quant_tensors,
      os.path.join(OUTPUT_FIXED_DIR, shard_name),
      metadata={"format": "pt"},
  )


else:
  # Multi-shard handling: load index, update shard mappings
  index_path = os.path.join(OUTPUT_FIXED_DIR, "model.safetensors.index.json")
  with open(index_path, "r") as f:
    index_data = json.load(f)


  weight_map = index_data["weight_map"]


  # Identify target shard where MTP was located
  mtp_shards = set(v for k, v in weight_map.items() if "mtp" in k.lower())
  target_shard = list(mtp_shards)[0] if mtp_shards else quant_safetensors[0]


  # Remove old quantized MTP references from weight map
  weight_map = {k: v for k, v in weight_map.items() if "mtp" not in k.lower()}


  # Point base MTP keys to target shard
  for k in base_mtp_tensors.keys():
    weight_map[k] = target_shard


  index_data["weight_map"] = weight_map
  with open(index_path, "w") as f:
    json.dump(index_data, f, indent=2)


  # Re-save shards
  for shard in quant_safetensors:
    shard_tensors = {}
    with safe_open(
        os.path.join(LOCAL_QUANT_DIR, shard), framework="pt", device="cpu"
    ) as f:
      for k in f.keys():
        if "mtp" not in k.lower():
          shard_tensors[k] = f.get_tensor(k)


    if shard == target_shard:
      shard_tensors.update(base_mtp_tensors)


    save_file(
        shard_tensors,
        os.path.join(OUTPUT_FIXED_DIR, shard),
        metadata={"format": "pt"},
    )


# -------------------------------------------------------------
# 4. Patch config.json for vLLM loader compatibility
# -------------------------------------------------------------
print("\nPatching config.json directives...")
config_path = os.path.join(OUTPUT_FIXED_DIR, "config.json")


with open(config_path, "r") as f:
  config = json.load(f)


q_cfg = config.get("quantization_config", {})
q_cfg["block_name_to_quantize"] = "model.language_model.layers"


# Ensure visual tower, mtp drafter, and lm_head are excluded
modules_to_exclude = {"mtp", "visual", "lm_head"}
existing_modules = set(q_cfg.get("modules_to_not_convert", []))
q_cfg["modules_to_not_convert"] = list(
    existing_modules.union(modules_to_exclude)
)


if "dynamic" not in q_cfg:
  q_cfg["dynamic"] = {}
q_cfg["dynamic"]["-:.*mtp.*"] = {}


if "extra_config" not in q_cfg:
  q_cfg["extra_config"] = {}
q_cfg["extra_config"][".*mtp.*"] = {"bits": 16, "data_type": "fp"}


config["quantization_config"] = q_cfg


with open(config_path, "w") as f:
  json.dump(config, f, indent=2)


# -------------------------------------------------------------
# 5. Push patched model to Hugging Face
# -------------------------------------------------------------
print(f"\nPushing fixed model to Hugging Face: {NEW_REPO_ID}...")
api = HfApi()
create_repo(
    NEW_REPO_ID, repo_type="model", exist_ok=True, private=False, token=HF_TOKEN
)
api.upload_folder(
    folder_path=OUTPUT_FIXED_DIR,
    repo_id=NEW_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
)


print(
    f"\n✅ Done! Fixed model uploaded to: https://huggingface.co/{NEW_REPO_ID}"
)

