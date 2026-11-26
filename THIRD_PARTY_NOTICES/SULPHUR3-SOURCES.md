# Sulphur-3 source and model notice

Source repository: [SulphurAI/Sulphur-3](https://huggingface.co/SulphurAI/Sulphur-3).
Pinned revision: `988224b9913740d60ee54b5a53c7a892467edc76` (inspected 2026-11-26).

The repository's optional `S3_ComfyUI.zip` describes a 55-block corrector and a post-CFG application node. T8 independently implements the public architecture/state-dict metadata and velocity-residual interface; upstream Python files and model weights are not bundled in this project. The original package is only used externally in differential validation.

|Upstream artifact|SHA256|
|---|---|
|README.md|`50b43f2a1c3c36e2577c256f2e5276fcff3ea9621543e50cca11ba3849441c1e`|
|LICENSE|`a017170627764c49e5890e1bbf0f7316e52db89e09b8b04ed92e4350178a1391`|
|S3_ComfyUI.zip|`7796a529e6d8bc663453821494b4135c4a7dec38ef6d54958d265b0437d5d5b5`|
|s3_corrector.safetensors|`7325543253724dd691884e88dbb6d363363f8e1f7eced4ca9ca27fa644913869`|

SulphurAI identifies the model as a MiniMax H3 derivative under the MiniMax H3 Community License. A copy of that model license is retained as [SULPHUR3-MODEL-LICENSE.txt](SULPHUR3-MODEL-LICENSE.txt); it does not relicense the model weights as this project's GPL code. Trailing whitespace in the retained license copy is normalized; the table hashes refer to the original upstream artifacts.

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

The original node IDs are `LoadH3Corrector` and `ApplyH3Corrector`; T8 appends distinct IDs and retains the `H3_CORRECTOR` socket type. No official endorsement is implied. The archive's experimental Motion/STG package is separate and is not included or represented as tested by this integration.
