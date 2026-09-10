"""MNIST loader with two interchangeable sources.

`idx` (default) fetches the four original idx archives from the S3 mirror and
parses the binary format by hand. `npz` fetches a single pre-packed bundle and
lets NumPy do the work. Both return identical arrays; `python data/mnist.py`
downloads both and asserts they agree.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).resolve().parent / "raw"

IDX_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
IDX_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
NPZ_KEYS = {
    "train_images": "x_train",
    "train_labels": "y_train",
    "test_images": "x_test",
    "test_labels": "y_test",
}


def _download(url: str, dest: Path) -> Path:
    """Fetch `url` to `dest` unless it is already cached. Writes atomically."""
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.parent / (dest.name + ".part")

    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle)

    partial.replace(dest)
    return dest


def _parse_idx(raw: bytes) -> np.ndarray:
    """Decode the idx binary format.

    Layout is a 4-byte magic number followed by one big-endian 4-byte dimension
    per axis, then the payload. Inside the magic number, byte 3 is a type code
    and byte 4 is the number of dimensions, so images (3 axes) and labels
    (1 axis) both fall out of the same code path.
    """
    type_code = raw[2]
    n_dims = raw[3]

    if type_code != 0x08:
        raise ValueError(f"expected unsigned-byte payload, got type code {type_code:#04x}")

    header_size = 4 + 4 * n_dims
    shape = np.frombuffer(raw[4:header_size], dtype=">u4").astype(int)

    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape)


def _load_idx() -> dict[str, np.ndarray]:
    arrays = {}
    for key, filename in IDX_FILES.items():
        path = _download(IDX_BASE + filename, RAW_DIR / filename)
        arrays[key] = _parse_idx(gzip.decompress(path.read_bytes()))
    return arrays


def _load_npz() -> dict[str, np.ndarray]:
    path = _download(NPZ_URL, RAW_DIR / "mnist.npz")
    with np.load(path) as bundle:
        return {key: bundle[npz_key] for key, npz_key in NPZ_KEYS.items()}


def _validate(arrays: dict[str, np.ndarray]) -> None:
    expected = {
        "train_images": (60000, 28, 28),
        "train_labels": (60000,),
        "test_images": (10000, 28, 28),
        "test_labels": (10000,),
    }

    for key, shape in expected.items():
        array = arrays[key]
        assert array.shape == shape, f"{key}: expected shape {shape}, got {array.shape}"
        assert array.dtype == np.uint8, f"{key}: expected uint8, got {array.dtype}"

    for key in ("train_labels", "test_labels"):
        assert set(np.unique(arrays[key])) == set(range(10)), f"{key}: labels are not 0-9"


def load_mnist(source: str = "idx") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train images, train labels, test images, test labels.

    Images are uint8 of shape (n, 28, 28) spanning 0-255; labels are uint8 0-9.
    Downloads are cached under `data/raw/`, so only the first call hits network.
    """
    loaders = {"idx": _load_idx, "npz": _load_npz}

    if source not in loaders:
        raise ValueError(f"unknown source {source!r}, expected one of {sorted(loaders)}")

    arrays = loaders[source]()
    _validate(arrays)

    return (
        arrays["train_images"],
        arrays["train_labels"],
        arrays["test_images"],
        arrays["test_labels"],
    )


if __name__ == "__main__":
    from_idx = _load_idx()
    _validate(from_idx)

    from_npz = _load_npz()
    _validate(from_npz)

    for key in from_idx:
        assert np.array_equal(from_idx[key], from_npz[key]), f"{key}: sources disagree"

    print("both sources validated and identical")
