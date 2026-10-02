"""Publish this reviewed, allowlisted Space bundle using the existing HF login."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = Path(__file__).resolve().parent
FILES = ["app.py", "engine.py", "motifs.py", "outputs.py", "verification.py", "retries.py", "parse_melody.js", "packages.txt", "prepare.py", "deploy.py", "tests/test_demo.py", "tests/test_verification.py", "tests/test_playback.py", "tests/editor_server.py", "tests/browser_editor.py", "requirements.txt", "README.md", "LICENSE",
         "checkpoint.json", "model.safetensors.gz", "assets/prompts.txt", "assets/motif-editor.html", "assets/motif-editor.css", "assets/motif-editor.js", "assets/abcjs-basic-min.js",
         "assets/vexflow-bravura-5.0.0.js", "assets/VEXFLOW-LICENSE", "assets/BRAVURA-LICENSE", "assets/ACADEMICO-LICENSE",
         "vendor/abc2xml.py", "vendor/notagen_core/__init__.py", "vendor/notagen_core/model.py",
         "vendor/notagen_core/backbones.py"]


def deploy(repo, private=False):
    for name in FILES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Missing {name}; run prepare.py first.")
    manifest = json.loads((ROOT / "checkpoint.json").read_text())
    digest = hashlib.sha256()
    with (ROOT / manifest["export_filename"]).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise ValueError("Exported weights do not match the checkpoint manifest.")
    api = HfApi()
    # The requested deployment must use a new repository. A name collision must
    # abort here, before create_commit can change any existing Space.
    api.create_repo(repo, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", private=private, exist_ok=False)
    info = api.space_info(repo)
    if info.private != private:
        raise ValueError("The new Space has unexpected visibility. Review its settings before uploading.")
    result = api.create_commit(repo, repo_type="space", operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(ROOT / name)) for name in FILES],
                               commit_message=f"Deploy MotiGen with checkpoint {manifest['source_filename']} (saved {manifest['source_modified_utc'][:10]}), live motif generation, scores and playback")
    print(result.commit_url)
    print(f"https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true", help="Keep the app, source, and weights accessible only to the owner")
    args = parser.parse_args()
    deploy(args.repo, private=args.private)
