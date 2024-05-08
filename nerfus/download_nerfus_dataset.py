"""
Downloads the Ultrasound in-the-wild dataset.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gdown
import tyro
from huggingface_hub import hf_hub_download
from nerfstudio.configs.base_config import PrintableConfig
from rich.console import Console
from typing_extensions import Annotated

CONSOLE = Console(width=120)


@dataclass
class DatasetDownload(PrintableConfig):
    """Download a dataset"""

    capture_name = None

    save_dir: Path = Path("data/")
    """The directory to save the dataset to"""

    def download(self, save_dir: Path) -> None:
        """Download the dataset"""
        raise NotImplementedError


@dataclass
class NerfusDataDownload(DatasetDownload):
    """Download the Nerfus dataset."""

    def download(self, save_dir: Path, hf=False):
        # Download the files
        if hf:
            download_path = hf_hub_download(
                repo_id="rishitdagli/us-in-the-wild",
                filename="dataset.zip",
                cache_dir="./",
                repo_type="dataset",
            )
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(str(save_dir))
            os.remove(download_path)
        else:
            url = "https://drive.google.com/uc?id=1m50igUELCYz5jnTks-fGpm_lNfcQPpYx"
            download_path = str(save_dir / "nerfus-dataset.zip")
            gdown.download(url, output=download_path)
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(str(save_dir))
            os.remove(download_path)


def main(
    dataset: DatasetDownload,
):
    """Script to download the Nerfus data.
    - captures: These are the videos which were used the ns-process-data.
    - data: These are the already-processed results from running ns-process-data (ie COLMAP).
    Args:
        dataset: The dataset to download (from).
    """
    dataset.save_dir.mkdir(parents=True, exist_ok=True)

    dataset.download(dataset.save_dir)


Commands = Union[Annotated[NerfusDataDownload, tyro.conf.subcommand(name="dataset")],]


def entrypoint():
    """Entrypoint."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()
