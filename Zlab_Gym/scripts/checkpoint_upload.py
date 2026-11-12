"""Optional Hugging Face mirroring for rank-zero training checkpoints."""

from concurrent.futures import Future, ThreadPoolExecutor
import os
from pathlib import Path


class HuggingFaceCheckpoints:
    def __init__(self, repo_id: str, run_name: str, api):
        self.repo_id = repo_id
        self.run_name = run_name
        self._api = api
        self._repo_ready = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-upload")
        self._pending: Future[None] | None = None

    @classmethod
    def from_environment(cls, run_name: str):
        repo_id = os.environ.get("ZLAB_HF_REPO_ID", "").strip()
        if not repo_id:
            return None

        try:
            from huggingface_hub import HfApi, get_token
            from huggingface_hub.utils import validate_repo_id
        except ImportError as exc:
            raise RuntimeError("Install the Hugging Face extra with: python -m pip install -e '.[huggingface]'") from exc

        validate_repo_id(repo_id)
        if "/" not in repo_id:
            raise ValueError("Set ZLAB_HF_REPO_ID to the model repository as owner/name.")
        token = get_token()
        if not token:
            raise RuntimeError("Set HF_TOKEN from your cloud secret store or log in to Hugging Face.")

        print(f"[HF] Checkpoints will be mirrored to {repo_id}/checkpoints/{run_name}/", flush=True)
        return cls(repo_id, run_name, HfApi(token=token))

    def submit(self, checkpoint: Path) -> None:
        # Allow one upload in flight, including when a run saves very frequently.
        if self._pending is not None:
            self._pending.result()
        # The runner may rewrite the final checkpoint at the same path.
        snapshot = checkpoint.read_bytes()
        self._pending = self._executor.submit(self._upload, checkpoint, snapshot)

    def _upload(self, checkpoint: Path, snapshot: bytes) -> None:
        destination = f"checkpoints/{self.run_name}/{checkpoint.name}"
        try:
            if not self._repo_ready:
                self._api.create_repo(self.repo_id, repo_type="model", private=True, exist_ok=True)
                self._repo_ready = True
            self._api.upload_file(
                path_or_fileobj=snapshot,
                path_in_repo=destination,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Save {self.run_name}/{checkpoint.name}",
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            print(f"[HF] Upload failed for {destination} ({detail}). Local file: {checkpoint}", flush=True)
        else:
            print(f"[HF] Uploaded {self.repo_id}/{destination}", flush=True)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
