# all that contain lifecycle_stage: deleted delete their parent folder
import os
import yaml
import shutil

MLRUNS_ROOT = r"mlruns"   # <-- change this to your mlruns directory
ASK_CONFIRMATION = True            # set to False if you want automatic deletion

def is_deleted_run(meta_path: str) -> bool:
    """Return True if the meta.yml indicates lifecycle_stage: deleted."""
    try:
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        return meta.get("lifecycle_stage") == "deleted"
    except Exception as e:
        print(f"Could not read {meta_path}: {e}")
        return False


def find_deleted_run_dirs(root: str):
    """Yield (expdir, run_id_dir) tuples for deleted runs."""
    for exp in os.listdir(root):
        exp_path = os.path.join(root, exp)
        if not os.path.isdir(exp_path):
            continue
        
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            if not os.path.isdir(run_path):
                continue

            meta_path = os.path.join(run_path, "meta.yaml")
            if os.path.isfile(meta_path) and is_deleted_run(meta_path):
                yield run_path


def main():
    deleted_dirs = list(find_deleted_run_dirs(MLRUNS_ROOT))

    if not deleted_dirs:
        print("No deleted runs found.")
        return

    print("The following run directories will be deleted:")
    for d in deleted_dirs:
        print("  ", d)

    if ASK_CONFIRMATION:
        confirm = input("\nProceed with deletion? (y/N): ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    for d in deleted_dirs:
        print(f"Deleting {d} ...")
        shutil.rmtree(d, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
