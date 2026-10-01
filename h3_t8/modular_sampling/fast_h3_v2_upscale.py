"""Source-bound, opt-in learned 3D handoff call for the V2 split pilot.

The existing learned upscaler remains the only numerical implementation. This
wrapper records its actual input/output and unmodified report in one call.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid

from .. import core, learned_latent_upscale_advanced, nodes_learned_latent_upscale_advanced
from ..learned_latent_upscale_advanced import learned_upscale_h3_av_latent
from .results import _input_identity, canonical


SCHEMA = "t8.modular-sampling.fasth3-v2-upscale-provenance.v1"
SETTINGS = ("model_name", "size_mode", "scale_by", "target_megapixels", "target_width",
            "target_height", "aspect_policy", "max_anisotropy", "precision", "release_policy")


def implementation_sha256():
    sources = {"v2_upscale_provenance": __file__,
               "learned_upscale": learned_latent_upscale_advanced.__file__,
               "learned_upscale_node": nodes_learned_latent_upscale_advanced.__file__,
               "core_geometry": core.__file__}
    files = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
             for name, path in sources.items()}
    return hashlib.sha256(canonical(files).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FastH3V2UpscaleReceipt:
    payload_json: str
    sha256: str

    def verify(self):
        payload = json.loads(self.payload_json)
        if payload.get("schema") != SCHEMA or canonical(payload) != self.payload_json:
            raise ValueError("Unknown or noncanonical V2 learned-upscale provenance")
        if hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("V2 learned-upscale provenance fingerprint changed")
        return payload


def upscale_with_provenance(av_latent, **kwargs):
    try:
        source = _input_identity(av_latent)
        settings = {name: kwargs[name] for name in SETTINGS}
        portable = True
    except Exception as error:
        source = {"unverified": f"{type(error).__name__}: {error}",
                  "execution_nonce": uuid.uuid4().hex}
        settings = None
        portable = False
    # No extra inference: the user-selected original function executes once.
    output, width, height, report_json = learned_upscale_h3_av_latent(
        latent=av_latent, **kwargs)
    try:
        output_identity = _input_identity(output)
        if not isinstance(report_json, str):
            raise ValueError("Learned upscale report is not JSON text")
        report_sha = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
    except Exception as error:
        output_identity = {"unverified": f"{type(error).__name__}: {error}",
                           "execution_nonce": uuid.uuid4().hex}
        report_sha = None
        portable = False
    payload = {"schema": SCHEMA, "implementation_sha256": implementation_sha256(),
               "portable_identity": portable,
               "input": source, "settings": settings,
               "output": output_identity, "width": width, "height": height,
               "report_sha256": report_sha,
               "boundary": "One learned 3D call only; no HIGH reconcile, parent acceptance or assembly"}
    encoded = canonical(payload)
    receipt = FastH3V2UpscaleReceipt(encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    receipt.verify()
    return output, width, height, report_json, receipt
