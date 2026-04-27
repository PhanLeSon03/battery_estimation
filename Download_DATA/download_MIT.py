from pathlib import Path
import shutil

import kagglehub


DATASET = "rickandjoe/mit-battery-degradation-dataset"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "DATA"


def dataset_already_local() -> bool:
    return DATA_DIR.exists() and any(DATA_DIR.iterdir())


def move_cache_to_data(cache_path: Path) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    for item in cache_path.iterdir():
        destination = DATA_DIR / item.name
        if destination.exists():
            continue
        shutil.move(str(item), str(destination))


if dataset_already_local():
    print("Path to dataset files:", DATA_DIR)
else:
    cache_path = Path(kagglehub.dataset_download(DATASET))
    move_cache_to_data(cache_path)
    print("Path to dataset files:", DATA_DIR)
