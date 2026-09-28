"""Upload agent trajectories + labels to the ASSERT-KTH/latent-programming-horizons-trajs
Hugging Face dataset.

Stages generations/<task>/<run_id>/ and labels/<task>/<run_id>/ into a symlink
tree matching the desired repo layout:

    <task>/<run_id>/<instance_id>[_runNN].json
    <task>/<run_id>/labels/<instance_id>[_runNN]_labels.json

then uploads the whole tree with `upload_large_folder`, which is built for
folders with thousands of files: it hashes/pre-uploads in parallel, commits in
small batches, and resumes via `.cache/.huggingface/` metadata inside the
staging dir if interrupted (a plain `upload_folder` commit of the full tree
times out server-side).

Usage:
    uv run python scripts/upload_trajectories_hf.py [--dry-run]
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "ASSERT-KTH/latent-programming-horizons-trajs"

RUNS = [
    ("swebench", "laguna_xs2_full"),
    ("swebench", "qwen36_35b_a3b_full"),
    ("swebench_pro", "laguna_xs2_full"),
    ("swebench_pro", "qwen36_35b_a3b_full"),
]


def build_staging_tree(stage_dir: Path, generations_dir: str, labels_dir: str) -> None:
    for task, run_id in RUNS:
        gen_dest = stage_dir / task / run_id
        gen_dest.mkdir(parents=True, exist_ok=True)
        for f in Path(f"{generations_dir}/{task}/{run_id}").glob("*.json"):
            link = gen_dest / f.name
            if not link.exists():
                link.symlink_to(f.resolve())

        label_dest = gen_dest / "labels"
        label_dest.mkdir(parents=True, exist_ok=True)
        for f in Path(f"{labels_dir}/{task}/{run_id}").glob("*.json"):
            link = label_dest / f.name
            if not link.exists():
                link.symlink_to(f.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="List files and total size without uploading.")
    parser.add_argument("--generations-dir", default="generations")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--dataset-card", default="README_dataset.md")
    parser.add_argument("--stage-dir", default=".hf_upload_stage")
    args = parser.parse_args()

    if args.dry_run:
        total_bytes = 0
        total_files = 0
        for task, run_id in RUNS:
            for base, dest_suffix in [(args.generations_dir, ""), (args.labels_dir, "/labels")]:
                folder = f"{base}/{task}/{run_id}"
                files = list(Path(folder).rglob("*.json"))
                size = sum(f.stat().st_size for f in files)
                total_bytes += size
                total_files += len(files)
                print(f"{folder} -> {task}/{run_id}{dest_suffix}  "
                      f"({len(files)} files, {size / 1e9:.2f} GB)")
        print(f"\nTotal: {total_files} files, {total_bytes / 1e9:.2f} GB")
        return

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)

    stage_dir = Path(args.stage_dir)
    build_staging_tree(stage_dir, args.generations_dir, args.labels_dir)

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(stage_dir),
        allow_patterns=["*.json"],
    )

    api.upload_file(
        repo_id=REPO_ID,
        repo_type="dataset",
        path_or_fileobj=args.dataset_card,
        path_in_repo="README.md",
        commit_message="Update dataset card",
    )


if __name__ == "__main__":
    main()
