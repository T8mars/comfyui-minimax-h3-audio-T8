"""Content/metadata-bound user presets. Filenames and architecture alone never select alpha."""
import json

from .semantic_bridge import SHAPES, _read_bridge_bytes, _weight_metadata
from .semantic_bridge_trans import contract_settings, spec_from_metadata, validate_contract

WUSHU_V1_SHA = "4b7a459aac43e066b9cd8d3f84f52b7d6c2777e7de05f35a3cb042abab889ebd"
COMIC_SHA = "d4303d77e1ff96b678484d891308eeeb261b919d9e6498610a2bb89f7aeaa9a9"
LEGACY_SETTINGS = {"alpha": .10, "magnitude_match": "per_token",
                   "token_scope": "all_tokens", "chunk_tokens": 256}


def inspect_bridge_profile(path, expected_sha=None):
    data, sha = _read_bridge_bytes(path, expected_sha)
    metadata = _weight_metadata(data)
    header_size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + header_size])
    keys = set(header) - {"__metadata__"}
    settings = None
    contract = None
    name = "Unidentified trainer Bridge"
    source = "manual_required"
    if keys == set(SHAPES):
        for key, shape in SHAPES.items():
            if header[key].get("shape") != list(shape) or header[key].get("dtype") not in ("F16", "BF16", "F32"):
                raise ValueError("Invalid six-tensor Bridge header")
        settings, name, source = dict(LEGACY_SETTINGS), "Original/BUNNY six-tensor MLP", "legacy_defaults"
        arch = "legacy_mlp"
    elif "net.in_proj.weight" in keys:
        spec = spec_from_metadata(metadata)
        contract, arch = spec.application_contract, "trans"
    elif "net.fc1.weight" in keys and metadata.get("arch") == "mlp":
        extra = json.loads(metadata.get("extra_json", "{}"))
        if not isinstance(extra, dict):
            raise ValueError("Invalid trainer Bridge extra_json")
        contract, arch = extra.get("application_contract"), "trainer_mlp"
    else:
        raise ValueError("Unsupported Semantic Bridge weight format; do not select a LoRA, JEV or H3 checkpoint")
    if contract is not None:
        if not isinstance(contract, dict):
            raise ValueError("Invalid Bridge application contract")
        settings = contract_settings(contract)
        source, name = "fixed_model_metadata", "Trainer Bridge (fixed application contract)"
    if sha == COMIC_SHA:
        name = "T8 comic-combat"
        if contract is None:
            raise ValueError("Known comic-combat content is missing its fixed contract")
    elif sha == WUSHU_V1_SHA:
        name, source = "Wushu v1", "sha_pinned_author_recommendation"
        settings = {"alpha": .12, "magnitude_match": "per_token",
                    "token_scope": "all_tokens", "chunk_tokens": 0}
    return {"name": name, "architecture": arch, "sha256": sha, "settings": settings,
            "settings_source": source, "fixed_contract": contract,
            "note": "Presets are inference settings, not video/voice quality acceptance; no JEV guard or auto-alpha."}


def validate_profile_settings(profile, *, alpha, magnitude_match, token_scope, chunk_tokens):
    validate_contract(profile["fixed_contract"], alpha=alpha, magnitude_match=magnitude_match,
                      token_scope=token_scope, chunk_tokens=chunk_tokens)
    actual = {"alpha": alpha, "magnitude_match": magnitude_match,
              "token_scope": token_scope, "chunk_tokens": chunk_tokens}
    settings = profile["settings"]
    return (["Manual settings differ from this weight's recommended preset; use the automatic config "
             "for the preset. Transformer chunking changes cross-token context."]
            if settings is not None and settings != actual else [])
