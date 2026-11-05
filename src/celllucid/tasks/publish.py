"""Copy an exported topic bank into assets/topic_banks/<name>, the folder that is published."""

from pathlib import Path
import argparse
import shutil
from .topic_bank import FILES, TopicBank


def promote_topic_bank(source, name, assets_dir="assets"):
    bank = TopicBank(source)
    target = Path(assets_dir).expanduser().resolve() / "topic_banks" / name
    target.mkdir(parents=True, exist_ok=True)
    # FILES ends with the manifest, so an interrupted copy cannot validate as a new bank.
    for file in FILES:
        shutil.copy2(bank.path / file, target / file)
    TopicBank(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, help="Exported topic-bank folder")
    parser.add_argument("--name", required=True, help="Bank name under assets/topic_banks, e.g. default or all_genes")
    parser.add_argument("--assets-dir", default="assets")
    args = parser.parse_args()
    print(promote_topic_bank(args.bank, args.name, args.assets_dir))


if __name__ == "__main__":
    main()
