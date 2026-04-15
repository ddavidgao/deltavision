"""
Perceptual hashing for coarse frame similarity.
Uses average hash (aHash) for speed. For higher accuracy, the imagehash
library provides DCT-based pHash — swap in via USE_IMAGEHASH flag.
"""

import numpy as np
from PIL import Image

USE_IMAGEHASH = False

try:
    import imagehash

    USE_IMAGEHASH = True
except ImportError:
    pass


def compute_phash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    """
    Compute perceptual hash.
    If imagehash is installed: DCT-based pHash (more robust).
    Otherwise: average hash (fast, adequate for transition detection).
    """
    if USE_IMAGEHASH:
        h = imagehash.phash(image, hash_size=hash_size)
        return h.hash.flatten()

    # Fallback: average hash
    resized = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    mean = arr.mean()
    return (arr > mean).flatten()


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Bitwise Hamming distance between two hashes."""
    return int(np.count_nonzero(h1 != h2))
