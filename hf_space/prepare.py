"""Export just the latest checkpoint's model weights; never upload training data or secrets."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = Path("/usr/xtmp/cy232/accompgen/weights")


def prepare(source=None):
    if source is None:
        candidates = list(DEFAULT_WEIGHTS.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints in {DEFAULT_WEIGHTS}")
        source = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    source = Path(source)
    before = source.stat()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    # GPU inference already uses fp16. Store that same precision and compress it
    # losslessly to fit a private Space's repository size limit.
    torch.set_num_threads(4)
    state = {key: value.to(torch.float16).contiguous().clone() for key, value in checkpoint["model"].items()}
    destination = ROOT / "model.safetensors.gz"
    with tempfile.TemporaryDirectory(prefix="motigen-export-") as temporary:
        uncompressed = Path(temporary) / "model.safetensors"
        save_file(state, str(uncompressed))
        with uncompressed.open("rb") as source_file, destination.open("wb") as target_file:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target_file, compresslevel=6, mtime=0) as compressed:
                shutil.copyfileobj(source_file, compressed, length=8 * 1024 * 1024)
    if source.stat().st_mtime_ns != before.st_mtime_ns:
        destination.unlink()
        raise RuntimeError("Checkpoint changed during export; retry after the save finishes.")
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "source_filename": source.name,
        "source_modified_utc": datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
        "epoch": checkpoint.get("epoch"), "phase": checkpoint.get("phase"),
        "best_epoch": checkpoint.get("best_epoch"), "min_eval_loss": checkpoint.get("min_eval_loss"),
        "sha256": digest.hexdigest(), "export_bytes": destination.stat().st_size,
        "export_filename": destination.name, "export_dtype": "float16", "compression": "gzip",
        "architecture": dict(encoder_backbone="gpt2", decoder_backbone="gpt2", patch_num_layers=20,
                             char_num_layers=6, hidden_size=1280, patch_length=1024, patch_size=16,
                             motif_attention_bias=4.0, patch_sampling_batch_size=0),
    }
    (ROOT / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    prepare(parser.parse_args().checkpoint)
