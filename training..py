!pip uninstall -y unsloth transformers peft trl wandb bitsandbytes accelerate datasets huggingface_hub tokenizers torch xformers || true
!pip install -U pip setuptools wheel
!pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-cache-dir --prefer-binary "torch==2.1.2" "triton==2.1.0" "xformers==0.0.23.post1" --index-url https://download.pytorch.org/whl/cu121
!pip install --no-cache-dir datasets huggingface_hub bitsandbytes accelerate sentencepiece safetensors

!pip install -U pip setuptools wheel
!pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-cache-dir --prefer-binary "torch==2.1.2" "triton==2.1.0" "xformers==0.0.23.post1" --index-url https://download.pytorch.org/whl/cu121
!pip install --no-cache-dir datasets huggingface_hub bitsandbytes accelerate sentencepiece safetensors

!pip install --upgrade "torchvision>=0.25.0"

# ============================================================
# Phi-4 Mini Reasoning Finetuning
# Unsloth + streaming + packing + hourly redundant snapshots
# ETA printing + automatic resume + background uploads
# ============================================================

import os
# Force PyArrow and HF Datasets to use minimal system RAM caching
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"

import gc
import time
import json
import queue
import random
import shutil
import threading
from pathlib import Path
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import numpy as np
import datasets

# Limit HuggingFace datasets internal memory to 100MB to stop streaming bloat
datasets.config.IN_MEMORY_MAX_SIZE = 100 * 1024 * 1024 

# IMPORTANT: import Unsloth before any heavy HF trainer stack
import unsloth
from unsloth import FastLanguageModel

from datasets import load_dataset
from huggingface_hub import HfApi, login, snapshot_download
from transformers import AutoTokenizer
from peft import PeftModel

# ============================================================
# USER SETTINGS
# ============================================================

HF_TOKEN = "token"
HUB_REPO = "dschauhan08/phi4-mini-reasoning-finetuned"

MODEL_NAME = "microsoft/Phi-4-mini-reasoning"

OUTPUT_DIR = Path("./phi4_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 512
BATCH = 2                  # set to 1 if you hit OOM, 2 is faster if it fits
GRAD_ACCUM = 8
LR = 2e-4
TRAIN_STEPS = 5000

UPLOAD_INTERVAL = 3600     # 1 hour

SEED = 42

DATASET_SOURCES = [
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "Crownelius/Opus-4.6-Reasoning-3300x",
    "Roman1111111/claude-opus-4.6-10000x",
    "Roman1111111/gemini-3-pro-10000x-hard-high-reasoning",
    "Roman1111111/gemini-3.1-pro-hard-high-reasoning",
    ("nvidia/Nemotron-Terminal-Corpus", "dataset_adapters"),
    "ianncity/Hunter-Alpha-SFT-220000x",
    "yatin-superintelligence/White-Hat-Security-Agent-Prompts-600K",
    ("OpenResearcher/OpenResearcher-Dataset", "seed_42"),
    "artillerywu/DeepResearch-9K",
    "nick007x/github-code-2025",
    "codeparrot/github-code",
    ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Formal-Logic"),
    ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Code-Concepts"),
]

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# HUB LOGIN
# ============================================================

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

try:
    api.create_repo(HUB_REPO, private=True, exist_ok=True)
except Exception:
    pass

# ============================================================
# RESUME STATE
# ============================================================

def _snapshot_name_key(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return -1

def find_latest_snapshot_name(repo_id: str) -> Optional[str]:
    try:
        files = api.list_repo_files(repo_id)
        snap_names = sorted(
            {
                p.split("/")[1]
                for p in files
                if p.startswith("snapshots/") and len(p.split("/")) > 1 and p.split("/")[1].startswith("snap_")
            },
            key=_snapshot_name_key,
        )
        return snap_names[-1] if snap_names else None
    except Exception:
        return None

def download_latest_snapshot(repo_id: str) -> Optional[Path]:
    snap_name = find_latest_snapshot_name(repo_id)
    if not snap_name:
        return None

    local_root = OUTPUT_DIR / "resume_snapshot"
    local_root.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"snapshots/{snap_name}/*",
            local_dir=str(local_root),
            token=HF_TOKEN,
        )
        snap_dir = local_root / "snapshots" / snap_name
        return snap_dir if snap_dir.exists() else None
    except Exception as e:
        print("Resume download failed:", e)
        return None

resume_snapshot = download_latest_snapshot(HUB_REPO)

resume_state = {
    "global_step": 0,
    "micro_step": 0,
    "last_upload_ts": time.time(),
    "source_offsets": [0] * len(DATASET_SOURCES),
    "token_buffer": [],
    "pending_sequences": [],
}

if resume_snapshot and (resume_snapshot / "state.json").exists():
    try:
        loaded = json.loads((resume_snapshot / "state.json").read_text())
        resume_state.update(loaded)
        print("Loaded resume state from:", resume_snapshot)
    except Exception as e:
        print("Could not read state.json:", e)

source_offsets = list(resume_state.get("source_offsets", [0] * len(DATASET_SOURCES)))
if len(source_offsets) != len(DATASET_SOURCES):
    source_offsets = [0] * len(DATASET_SOURCES)

global_step = int(resume_state.get("global_step", 0))
micro_step = int(resume_state.get("micro_step", 0))
last_upload_ts = float(resume_state.get("last_upload_ts", time.time()))
resume_token_buffer = list(resume_state.get("token_buffer", []))
resume_pending_sequences = [list(x) for x in resume_state.get("pending_sequences", [])]

# ============================================================
# MODEL + TOKENIZER
# ============================================================

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=SEQ_LEN,
    load_in_4bit=True,
)

if resume_snapshot and (resume_snapshot / "tokenizer").exists():
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(resume_snapshot / "tokenizer"), use_fast=True)
    except Exception:
        pass

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if resume_snapshot and (resume_snapshot / "adapter").exists():
    print("Loading adapter from:", resume_snapshot / "adapter")
    model = PeftModel.from_pretrained(
        base_model,
        str(resume_snapshot / "adapter"),
        is_trainable=True,
    )
else:
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

try:
    FastLanguageModel.for_training(model)
except Exception:
    pass

try:
    model.config.use_cache = False
except Exception:
    pass

device = next(model.parameters()).device
trainable_params = [p for p in model.parameters() if p.requires_grad]
print("Trainable parameters:", sum(p.numel() for p in trainable_params))

optimizer = torch.optim.AdamW(trainable_params, lr=LR)

if resume_snapshot and (resume_snapshot / "optimizer.pt").exists():
    try:
        optimizer.load_state_dict(torch.load(resume_snapshot / "optimizer.pt", map_location="cpu"))
        print("Loaded optimizer state.")
    except Exception as e:
        print("Optimizer state load failed:", e)

autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

# ============================================================
# DATA HELPERS
# ============================================================

def normalize(example) -> str:
    if isinstance(example, dict):
        for key in ("text", "content", "prompt", "response", "completion"):
            if key in example and isinstance(example[key], str):
                txt = example[key].strip()
                if txt:
                    return txt
        if "messages" in example and isinstance(example["messages"], list):
            parts = []
            for m in example["messages"]:
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content = m.get("content", "") or m.get("text", "")
                    parts.append(f"{role}: {content}")
            if parts:
                return "\n".join(parts)
    return json.dumps(example, ensure_ascii=False, default=str)

def load_stream(spec, skip: int = 0):
    try:
        if isinstance(spec, tuple):
            ds = load_dataset(spec[0], spec[1], split="train", streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(spec, split="train", streaming=True, trust_remote_code=True)

        try:
            cols = ds.column_names
            if isinstance(cols, list):
                keep = {"text", "content", "prompt", "response", "completion", 
                        "messages", "conversations", "system", "instruction", 
                        "output", "input", "history", "dialogue", "question", "answer"}
                drop_cols = [c for c in cols if c.lower() not in keep]
                if drop_cols:
                    ds = ds.remove_columns(drop_cols)
        except Exception:
            pass

        it = iter(ds)
        for _ in range(skip):
            next(it, None)
        return it
    except Exception as e:
        print("Skipping:", spec, "|", e)
        return None

class PackedBatcher:
    def __init__(
        self,
        tokenizer,
        seq_len: int,
        batch_size: int,
        source_offsets: List[int],
        token_buffer: Optional[List[int]] = None,
        pending_sequences: Optional[List[List[int]]] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.source_offsets = list(source_offsets)
        self.token_buffer = list(token_buffer or [])
        self.pending_sequences = [list(x) for x in (pending_sequences or [])]
        self.streams = self._build_streams()
        self.text_iter = self._round_robin_text()

    def _build_streams(self):
        streams = []
        for idx, spec in enumerate(DATASET_SOURCES):
            it = load_stream(spec, skip=self.source_offsets[idx])
            if it is not None:
                streams.append([idx, spec, it])
                print("Loaded:", spec)
        return streams

    def _round_robin_text(self):
        active = self.streams.copy()
        while active:
            progressed = False
            for item in active[:]:
                idx, spec, it = item
                try:
                    ex = next(it)
                    self.source_offsets[idx] += 1
                    progressed = True
                    
                    txt = normalize(ex)
                    
                    # === THE SYSTEM RAM KILLER FIX ===
                    # Truncate strings to ~15,000 characters (~3000 tokens) BEFORE 
                    # they reach the Tokenizer. If a repo yields a 2-million char string, 
                    # the tokenizer will spike RAM by 8GB instantly if you don't do this.
                    # 15,000 chars is more than enough to perfectly pack a 512 seq_len.
                    if len(txt) > 15000:
                        txt = txt[:15000]
                        
                    yield txt
                    
                    # Force delete raw references so memory is freed immediately
                    del ex
                    del txt
                    
                except StopIteration:
                    active.remove(item)
                except Exception as e:
                    print("Stream error:", spec, "|", e)
                    active.remove(item)
                    
            if not progressed:
                break

    def next_batch(self):
        while len(self.token_buffer) >= self.seq_len and len(self.pending_sequences) < self.batch_size:
            self.pending_sequences.append(self.token_buffer[:self.seq_len])
            self.token_buffer = self.token_buffer[self.seq_len:]

        while len(self.pending_sequences) < self.batch_size:
            try:
                text = next(self.text_iter)
            except StopIteration:
                break

            ids = self.tokenizer(
                text, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=self.seq_len * 5 # Lower max token capacity
            )["input_ids"]
            
            del text # Free string memory

            if not ids:
                continue

            self.token_buffer.extend(ids)

            while len(self.token_buffer) >= self.seq_len:
                self.pending_sequences.append(self.token_buffer[:self.seq_len])
                self.token_buffer = self.token_buffer[self.seq_len:]

                if len(self.pending_sequences) >= self.batch_size:
                    break

        if not self.pending_sequences:
            return None

        sequences = self.pending_sequences[:self.batch_size]
        self.pending_sequences = self.pending_sequences[self.batch_size:]

        batch = torch.tensor(sequences, dtype=torch.long, device=device)
        attn = torch.ones_like(batch)

        return {"input_ids": batch, "attention_mask": attn}

    def state_dict(self):
        return {
            "source_offsets": self.source_offsets,
            "token_buffer": self.token_buffer,
            "pending_sequences": self.pending_sequences,
        }

batcher = PackedBatcher(
    tokenizer=tokenizer,
    seq_len=SEQ_LEN,
    batch_size=BATCH,
    source_offsets=source_offsets,
    token_buffer=resume_token_buffer,
    pending_sequences=resume_pending_sequences,
)

# ============================================================
# SNAPSHOT / UPLOAD
# ============================================================

upload_queue: "queue.Queue[Path]" = queue.Queue(maxsize=2)
stop_event = threading.Event()
state_lock = threading.Lock()
upload_lock = threading.Lock()

def prune_old_snapshots(keep: int = 3):
    root = OUTPUT_DIR / "snapshots"
    if not root.exists():
        return
    snaps = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("snap_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in snaps[keep:]:
        shutil.rmtree(old, ignore_errors=True)

def save_snapshot(reason: str):
    ts = int(time.time())
    snap_name = f"snap_{ts}"
    snap_dir = OUTPUT_DIR / "snapshots" / snap_name
    adapter_dir = snap_dir / "adapter"
    tokenizer_dir = snap_dir / "tokenizer"

    snap_dir.mkdir(parents=True, exist_ok=True)

    with state_lock:
        state = {
            "reason": reason,
            "timestamp": ts,
            "global_step": global_step,
            "micro_step": micro_step,
            "last_upload_ts": ts,
            **batcher.state_dict(),
            "seq_len": SEQ_LEN,
            "batch": BATCH,
            "grad_accum": GRAD_ACCUM,
        }

    try:
        model.save_pretrained(str(adapter_dir))
    except Exception as e:
        print("Adapter save failed:", e)

    try:
        tokenizer.save_pretrained(str(tokenizer_dir))
    except Exception:
        pass

    try:
        torch.save(optimizer.state_dict(), snap_dir / "optimizer.pt")
    except Exception as e:
        print("Optimizer save failed:", e)

    (snap_dir / "state.json").write_text(json.dumps(state, indent=2))

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    try:
        upload_queue.put_nowait(snap_dir)
        print("Queued snapshot for upload:", snap_name)
    except queue.Full:
        print("Upload queue full. Kept local snapshot:", snap_name)

def upload_worker():
    while True:
        if stop_event.is_set() and upload_queue.empty():
            break

        try:
            folder = upload_queue.get(timeout=5)
        except queue.Empty:
            continue

        try:
            api.upload_folder(
                repo_id=HUB_REPO,
                folder_path=str(folder),
                path_in_repo=f"snapshots/{folder.name}",
                token=HF_TOKEN,
            )
            print("Uploaded snapshot:", folder.name)
            prune_old_snapshots(keep=3)
        except Exception as e:
            print("Upload failed:", e)
        finally:
            upload_queue.task_done()

uploader = threading.Thread(target=upload_worker, daemon=True)
uploader.start()

# ============================================================
# TRAIN LOOP + ETA
# ============================================================

session_start = time.perf_counter()
start_global_step = global_step

print("Starting training from global_step:", global_step)

try:
    optimizer.zero_grad(set_to_none=True)

    while global_step < TRAIN_STEPS:
        batch = batcher.next_batch()
        if batch is None:
            print("Data stream ended.")
            break

        with autocast_ctx:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            loss = outputs.loss / GRAD_ACCUM

        loss.backward()
        micro_step += 1
        
        loss_val = loss.detach().float().item()
        
        del outputs
        del loss
        del batch

        if micro_step >= GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            micro_step = 0

            completed = max(1, global_step - start_global_step)
            elapsed = time.perf_counter() - session_start
            sec_per_step = elapsed / completed
            eta_seconds = sec_per_step * max(0, TRAIN_STEPS - global_step)

            if global_step % 10 == 0:
                print(
                    f"step {global_step}/{TRAIN_STEPS} "
                    f"loss {loss_val * GRAD_ACCUM:.4f} "
                    f"eta {eta_seconds/3600:.2f}h"
                )

            if time.time() - last_upload_ts >= UPLOAD_INTERVAL:
                print("Saving hourly snapshot...")
                save_snapshot("hourly")
                last_upload_ts = time.time()
                
            # Free unused Python memory loops routinely to keep memory low
            if global_step % 50 == 0:
                gc.collect()

    if micro_step > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        micro_step = 0

finally:
    print("Saving final snapshot...")
    save_snapshot("final")

    try:
        upload_queue.join()
    except Exception:
        pass

    stop_event.set()
    try:
        uploader.join(timeout=30)
    except Exception:
        pass

print("Done.")


import os
from huggingface_hub import HfApi, snapshot_download
import unsloth
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# ==========================================================
# 1. SETUP & AUTHENTICATION
# ==========================================================
HF_TOKEN = "" 
HUB_REPO = "dschauhan08/phi4-mini-reasoning-finetuned"
HUB_REPO_MERGED = "dschauhan08/phi4-mini-ultra-reasoning" 
HUB_REPO_GGUF = "dschauhan08/phi4-mini-ultra-reasoning-GGUF" 

# ==========================================================
# 2. AUTO-DOWNLOAD THE FINAL ADAPTER
# ==========================================================
print("Connecting to Hugging Face...")
api = HfApi(token=HF_TOKEN)

try:
    files = api.list_repo_files(HUB_REPO)
    snaps = sorted(list(set([f.split('/')[1] for f in files if f.startswith('snapshots/snap_')])))
    latest_snap = snaps[-1] # This will grab snap_1774026060
    print(f"Downloading final snapshot: {latest_snap}")

    snapshot_download(
        repo_id=HUB_REPO,
        allow_patterns=f"snapshots/{latest_snap}/*",
        token=HF_TOKEN,
        local_dir="./downloaded_weights"
    )
    FINAL_ADAPTER_DIR = f"./downloaded_weights/snapshots/{latest_snap}/adapter"
except Exception as e:
    print(f"Fetch Error: {e}")
    exit()

# ==========================================================
# 3. LOAD MODEL
# ==========================================================
print(f"\nLoading and fusing weights into RAM...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = FINAL_ADAPTER_DIR,
    max_seq_length = 512,
    dtype = None,  # Auto-detects the native 16-bit precision
    load_in_4bit = False, 
)

# ==========================================================
# 4. EXPORT STANDARD FULL-WEIGHT MODEL (Transformers Format)
# ==========================================================
print("\n--- Uploading standard full-weight Hugging Face model ---")
try:
    # merged_16bit is the native, highest-quality format for the base model.
    model.push_to_hub_merged(
        HUB_REPO_MERGED,
        tokenizer,
        save_method = "merged_16bit",
        token = HF_TOKEN,
    )
    print("Success! Full-weight merged model uploaded.")
except Exception as e:
    print(f"Failed to upload merged model: {e}")

# ==========================================================
# 5. EXPORT TO GGUF FORMATS
# ==========================================================
# I added f16 (Native uncompressed) and f32 (Padded uncompressed)
quantizations = ["f16", "f32", "fp8", "q8_0", "q4_k_m", "q4_k_s", "q2_k"]

print("\nStarting Sequential GGUF Build & Upload...")
for quant in quantizations:
    print(f"\n--- Processing: {quant} ---")
    try:
        model.push_to_hub_gguf(
            HUB_REPO_GGUF,
            tokenizer,
            quantization_method = quant,
            token = HF_TOKEN,
        )
        print(f"Success! {quant} uploaded.")
    except Exception as e:
        print(f"Failed on {quant}: {e}")

print("\nAll files successfully exported! You are ready to benchmark.")
