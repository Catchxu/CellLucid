"""Promote a trained topic bank to assets and explicitly publish it to the Hub."""

from pathlib import Path
import argparse
import os
import shutil
from ..assets import TOPIC_BANK_FILES
from .topic_bank import TopicBank


def promote_topic_bank(source, assets_dir="assets"):
    bank = TopicBank(source)
    target = Path(assets_dir).expanduser().resolve() / "topic_bank"
    if bank.path.resolve() == target:
        return target
    target.mkdir(parents=True, exist_ok=True)
    # Publish the manifest last so an interrupted copy cannot validate as a new bank.
    for name in TOPIC_BANK_FILES:
        shutil.copy2(bank.path / name, target / name)
    TopicBank(target)
    return target


def upload_topic_bank(path, repo_id, *, repo_type="space", commit_message="Upload CellLucid topic bank"):
    from huggingface_hub import HfApi

    bank = TopicBank(path)
    return HfApi().upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(bank.path.resolve()),
        path_in_repo="topic_bank",
        allow_patterns=list(TOPIC_BANK_FILES),
        commit_message=commit_message,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CKPT_DIR", "runs/hca_pretraining/topic_bank"), help="Completed, exported topic-bank folder")
    parser.add_argument("--assets-dir", default=os.environ.get("ASSETS_DIR", "assets"))
    parser.add_argument("--push-hf", type=int, choices=[0, 1], default=int(os.environ.get("PUSH_HF", "0")))
    parser.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--hf-repo-type", choices=["space", "model"], default=os.environ.get("HF_REPO_TYPE", "space"))
    parser.add_argument("--commit-message", default="Upload CellLucid topic bank")
    args = parser.parse_args()
    if args.push_hf and args.hf_repo_id is None:
        parser.error("--hf-repo-id is required with --push-hf 1")
    target = promote_topic_bank(args.checkpoint_dir, args.assets_dir)
    print(f"Promoted topic bank to {target}")
    if args.push_hf:
        result = upload_topic_bank(target, args.hf_repo_id, repo_type=args.hf_repo_type, commit_message=args.commit_message)
        print(result.commit_url)


if __name__ == "__main__":
    main()
