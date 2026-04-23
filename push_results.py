import json
from huggingface_hub import HfApi, create_repo, upload_file
import config

def push_daily_result(payload: dict):
    filename = f"neural_operator_{config.TODAY}.json"
    with open(filename, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved local file: {filename}")
    if config.HF_TOKEN:
        api = HfApi(token=config.HF_TOKEN)
        try:
            create_repo(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset", token=config.HF_TOKEN, exist_ok=True)
        except:
            pass
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=config.HF_OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.HF_OUTPUT_REPO}/{filename}")
