"""Explicitly freeze the new R6 source without installing or changing old configs.

Uses the existing exact Git-tree/blob archive verifier in this process only.
The original preparation tool and its old default revisions stay unchanged.
"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_freevideo_quality_source as source  # noqa: E402


REVISION = "f2c25e6e4f82d5ad9ed966998423762503d98e5c"


def main():
    if "--audio-reference-v3" in sys.argv:
        raise ValueError("This explicit source preparer freezes R6, not the old audio-v3 tree")
    previous = source.FREEVIDEO_REVISION
    try:
        source.FREEVIDEO_REVISION = REVISION
        source.main()
    finally:
        source.FREEVIDEO_REVISION = previous


if __name__ == "__main__":
    main()
