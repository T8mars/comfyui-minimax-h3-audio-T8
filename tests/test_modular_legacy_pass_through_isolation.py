"""Moving new serial nodes must restore frozen legacy bytes, not change behavior."""
import ast
import hashlib
from pathlib import Path

import pytest

from h3_audio_t8_pkg.modular_sampling import node_classes
from h3_audio_t8_pkg.modular_sampling import legacy_pass_through_nodes as serial
from h3_audio_t8_pkg.nodes_motion_recovery_advanced import (
    MiniMaxH3MotionOverloadAnalyzeT8Advanced,
)
from h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative,sha", [
    ("h3_t8/nodes_motion_recovery_advanced.py",
     "5757103ae90f939954245f2d6751507b02a82fa367d9b35ba04c5eb03f134ab7"),
    ("h3_t8/nodes_native_latent_checkpoint_advanced.py",
     "afe3571e0162714c5f4a969347f4daf29015d8d2e6a248d469687f6a1d1f0a7d"),
])
def test_both_old_modules_are_the_exact_frozen_m0_source(relative, sha):
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == sha


@pytest.mark.parametrize("name,sha", [
    ("MiniMaxH3MotionOverloadAnalyzePassThroughEXPT8",
     "2e9e56d64619e65648436117d87e26c6382d55654b99b23e4b556dd38734afbd"),
    ("MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8",
     "999838cf334350c3cdde2850965f8ea2f888a7b27b854842b2176ba6991ba770"),
])
def test_serial_class_body_is_byte_exact_to_the_prior_addition(name, sha):
    # Recorded by the pre-move source audit; its insertion includes three LFs.
    # This fixes the migration boundary without uploading private audit files.
    text = Path(serial.__file__).read_text(encoding="utf-8")
    classes = {node.name: node for node in ast.parse(text).body
               if isinstance(node, ast.ClassDef)}
    original_insertion = ast.get_source_segment(text, classes[name]) + "\n\n\n"
    assert hashlib.sha256(original_insertion.encode()).hexdigest() == sha


def test_serial_nodes_keep_original_parents_and_registration_order():
    checkpoint = serial.MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8
    motion = serial.MiniMaxH3MotionOverloadAnalyzePassThroughEXPT8
    assert checkpoint.__bases__ == (MiniMaxH3NativeLatentCheckpointSaveT8Advanced,)
    assert motion.__bases__ == (MiniMaxH3MotionOverloadAnalyzeT8Advanced,)
    assert node_classes()[-4:-2] == [checkpoint, motion]
    assert motion.execute.__func__ is MiniMaxH3MotionOverloadAnalyzeT8Advanced.execute.__func__
    assert motion.define_schema().is_output_node is False
    assert checkpoint.define_schema().is_output_node is False
    assert MiniMaxH3MotionOverloadAnalyzeT8Advanced.define_schema().is_output_node is True
    assert MiniMaxH3NativeLatentCheckpointSaveT8Advanced.define_schema().is_output_node is True
