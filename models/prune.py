import os
from huggingface_hub import HfApi, CommitOperationDelete

REPO = "Athithya117/dinghy-law-0.6b-v1-gguf"

# Keep: the Q8_0 model, plus the metadata files HF needs
KEEP = {
    "dinghy-law-0.6b-v1-Q8_0.gguf",
    ".gitattributes",
    "README.md",
    "gguf_fidelity.json",
}

api = HfApi(token=os.environ["HF_TOKEN"])
all_files = api.list_repo_files(repo_id=REPO, repo_type="model")
to_delete = [f for f in all_files if f not in KEEP]

print(f"Deleting: {to_delete}")
if to_delete:
    api.create_commit(
        repo_id=REPO,
        repo_type="model",
        operations=[CommitOperationDelete(path_in_repo=f) for f in to_delete],
        commit_message="Keep only Q8_0 quantization",
    )
    print("✅ Deleted.")
