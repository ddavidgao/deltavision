"""
Tests for perceptual hashing.
"""

import numpy as np
from PIL import Image
import pytest

from vision.phash import compute_phash, hamming_distance


def make_gradient(direction="horizontal", size=(200, 200)) -> Image.Image:
    """Create a gradient image — pHash needs texture to be meaningful."""
    w, h = size
    if direction == "horizontal":
        arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    else:
        arr = np.tile(np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1), (1, w))
    return Image.fromarray(arr)


def make_checkerboard(block_size=25, size=(200, 200)) -> Image.Image:
    """Create a checkerboard pattern."""
    w, h = size
    arr = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((x // block_size) + (y // block_size)) % 2 == 0:
                arr[y, x] = 255
    return Image.fromarray(arr)


class TestPHash:
    def test_identical_images_zero_distance(self):
        img = make_gradient()
        h1 = compute_phash(img)
        h2 = compute_phash(img)
        assert hamming_distance(h1, h2) == 0

    def test_completely_different_high_distance(self):
        """Structurally different images should have high hash distance."""
        horiz = make_gradient("horizontal")
        checker = make_checkerboard()
        h1 = compute_phash(horiz)
        h2 = compute_phash(checker)
        assert hamming_distance(h1, h2) > 10

    def test_similar_images_low_distance(self):
        """Slight modification should produce lower distance than completely different."""
        img1 = make_gradient("horizontal", size=(400, 400))
        arr2 = np.array(img1)
        arr2[195:205, 195:205] += 30  # small nudge in center
        np.clip(arr2, 0, 255, out=arr2)
        img2 = Image.fromarray(arr2.astype(np.uint8))
        h1 = compute_phash(img1)
        h2 = compute_phash(img2)
        # Should be closer than structurally different images
        different = make_checkerboard(block_size=50, size=(400, 400))
        h3 = compute_phash(different)
        assert hamming_distance(h1, h2) < hamming_distance(h1, h3)

    def test_hash_size(self):
        img = make_gradient()
        h = compute_phash(img, hash_size=8)
        assert len(h) == 64

        h16 = compute_phash(img, hash_size=16)
        assert len(h16) == 256
