import argparse
import numpy as np
from skimage import io, color, transform


def to_gray(image):
    """Convert image to grayscale, handling RGBA input."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        return color.rgb2gray(image)
    return image


def detect_change(before_path, after_path, threshold=0.2):
    """Compute a binary change mask from two images."""
    before = io.imread(before_path).astype(np.float32) / 255.0
    after = io.imread(after_path).astype(np.float32) / 255.0

    if before.shape != after.shape:
        after = transform.resize(after, before.shape, preserve_range=True)

    before_gray = to_gray(before)
    after_gray = to_gray(after)

    diff = np.abs(before_gray - after_gray)
    change_map = diff > threshold
    return change_map


def main():
    parser = argparse.ArgumentParser(description="Simple change detection")
    parser.add_argument("before", help="Image captured before change")
    parser.add_argument("after", help="Image captured after change")
    parser.add_argument("output", help="Path to save the change map")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Pixel difference threshold [0-1]")
    args = parser.parse_args()

    mask = detect_change(args.before, args.after, args.threshold)
    io.imsave(args.output, (mask * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
