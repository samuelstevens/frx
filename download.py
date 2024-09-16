import dataclasses
import os
import tarfile

import requests
import tqdm
import tyro

imagenet_v2_url = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"


@dataclasses.dataclass(frozen=True)
class Args:
    root: str = "."
    """where to download files."""

    imagenet_v2: bool = True
    """whether to download imagenet v2."""

    chunk_size_kb: int = 1
    """how many KB to download at a time before writing to file."""


def download_url(url: str, path: str, chunk_size: int):
    r = requests.get(url, stream=True)
    r.raise_for_status()

    n_bytes = int(r.headers["content-length"])

    t = tqdm.tqdm(
        total=n_bytes, unit="B", unit_scale=1, unit_divisor=1024, desc="Downloading"
    )
    with open(path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
            t.update(len(chunk))
    t.close()
    print(f"Downloaded: {path}.")


def main(args: Args):
    chunk_size = int(args.chunk_size_kb * 1024)

    if args.imagenet_v2:
        dirpath = os.path.join(args.root, "imagenetv2")
        os.makedirs(dirpath, exist_ok=True)
        tarpath = os.path.join(dirpath, "imagenetv2.tar.gz")
        download_url(imagenet_v2_url, tarpath, chunk_size)
        with tarfile.open(tarpath, "r") as tar:
            for member in tqdm.tqdm(tar, desc="Extracting images", total=11_001):
                tar.extract(member, path=dirpath, filter="data")
        print("Extracted imagenet-v2.")


if __name__ == "__main__":
    main(tyro.cli(Args))
