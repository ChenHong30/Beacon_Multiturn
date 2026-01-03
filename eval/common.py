import json
import logging
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_DEVNULL = None


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _open_devnull():
    global _DEVNULL
    if _DEVNULL is None:
        _DEVNULL = open(os.devnull, "w")
    return _DEVNULL


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_rank_from_env() -> Optional[int]:
    for name in (
        "RANK",
        "LOCAL_RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "MPI_RANK",
        "PMI_RANK",
    ):
        val = os.getenv(name)
        if not val:
            continue
        try:
            return int(val)
        except ValueError:
            continue
    return None


def is_main_process(rank: Optional[int] = None) -> bool:
    if rank is None:
        rank = get_rank_from_env()
    return rank in (None, 0)


def configure_process_logging(
    *,
    rank: Optional[int] = None,
    force_non_main: bool = False,
    suppress_stdout: bool = True,
    suppress_warnings: bool = True,
    log_all_env: str = "LOG_ALL_RANKS",
) -> None:
    if _str_to_bool(os.getenv(log_all_env, "")):
        return
    if not force_non_main and is_main_process(rank):
        return

    if suppress_warnings:
        warnings.filterwarnings("ignore")

    logging.getLogger().setLevel(logging.ERROR)
    for name in ("transformers", "datasets", "torch", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)

    if suppress_stdout:
        sys.stdout = _open_devnull()


def parse_cuda_ids(cuda_id: Any, cuda_ids: Optional[Any]) -> List[int]:
    source = cuda_ids if cuda_ids is not None else cuda_id
    if source is None:
        return [0]

    if isinstance(source, (list, tuple)):
        ids = [int(x) for x in source]
    elif isinstance(source, str):
        s = source.strip()
        if not s:
            ids = [0]
        else:
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        ids = [int(x) for x in parsed]
                    else:
                        ids = [0]
                except Exception:
                    ids = [int(x) for x in re.split(r"[,\s]+", s) if x]
            else:
                ids = [int(x) for x in re.split(r"[,\s]+", s) if x]
    else:
        ids = [int(source)]

    return list(dict.fromkeys(ids))


def validate_cuda_ids(cuda_ids: List[int]) -> None:
    import torch

    if not torch.cuda.is_available():
        if cuda_ids and cuda_ids != [0]:
            print(f"[WARN] CUDA not available; falling back to CPU (cuda_ids={cuda_ids}).")
        return

    device_count = torch.cuda.device_count()
    for cid in cuda_ids:
        if cid < 0 or cid >= device_count:
            raise ValueError(
                f"Invalid cuda_id={cid}; torch.cuda.device_count()={device_count}."
            )
