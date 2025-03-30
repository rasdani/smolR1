import os
import glob
from huggingface_hub import HfApi, create_repo, ModelCard, ModelCardData

api = HfApi()

local_checkpoints_folder = "./outputs/simple_rl_zoo_qwen2.5-0.5b"

card_data = ModelCardData(
    language="en",
    license="apache-2.0",
    model_name="Qwen2.5-0.5B-simpleRL-Zoo",
    base_model="Qwen/Qwen2.5-0.5B",
    tags=["reinforcement-learning", "qwen"],
)
card = ModelCard.from_template(
    card_data,
)
card.save("model_card.md")

repo_id = "rasdani/Qwen2.5-0.5B-simpleRL-Zoo"
repo_type = "model"

create_repo(repo_id, repo_type=repo_type, exist_ok=True)

# Find the latest checkpoint
checkpoint_pattern = os.path.join(local_checkpoints_folder, "checkpoint-*")
checkpoint_folders = sorted(
    [f for f in glob.glob(checkpoint_pattern) if os.path.isdir(f)],
    key=lambda folder: int(os.path.basename(folder).split('-')[-1])
)

if not checkpoint_folders:
    print("No checkpoints found.")
    exit()

last_checkpoint_path = checkpoint_folders[-1]
print(f"Last checkpoint identified as: {os.path.basename(last_checkpoint_path)}")

# Upload other checkpoints to subfolders
for local_path in checkpoint_folders[:-1]:
    checkpoint_name = os.path.basename(local_path)
    print(f"Uploading {checkpoint_name} to subfolder '{checkpoint_name}'...")
    try:
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=checkpoint_name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload checkpoint {checkpoint_name}"
        )
    except Exception as e:
        print(f"Error uploading subfolder {checkpoint_name}: {e}")

# Upload the last checkpoint to the root directory
last_checkpoint_name = os.path.basename(last_checkpoint_path)
print(f"Uploading last checkpoint {last_checkpoint_name} to root directory...")
api.upload_folder(
    folder_path=last_checkpoint_path,
    path_in_repo=".",
    repo_id=repo_id,
    repo_type=repo_type
)
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Add model card"
)
