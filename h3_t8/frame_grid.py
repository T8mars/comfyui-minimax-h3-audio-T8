"""Native frame-grid constants/math without Torch or Core imports."""

FPS = 24
MIN_TRAINED_FRAMES = 124


def align_frame_count(frame_count: int) -> int:
    """Snap up to MiniMax H3's 17n+5 frame grid."""
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)
