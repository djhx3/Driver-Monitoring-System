import os
from huggingface_hub import HfApi, HfFolder

# Hugging Face model repository (replace with your username)
REPO_ID = "djh3/driver-monitoring-model"

# Get Hugging Face API token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HF_TOKEN is None:
    raise ValueError("Hugging Face token not found. Set HUGGINGFACE_TOKEN in GitHub Secrets.")

# Login using the token
HfFolder.save_token(HF_TOKEN)
api = HfApi()

# Path to the trained model
MODEL_PATH = "best.pt"

# Upload model to Hugging Face
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="/training/best.pt",
    repo_id=REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)

print(f"✅ Model successfully uploaded to Hugging Face: https://huggingface.co/{REPO_ID}")
