# Pika H3 Flex reference sources

This integration supplies original T8 recipe glue around existing ComfyUI LCM
and FLOW_AV. No Pika model weights, inference implementation, or example images
are redistributed with it.

- Official inference repository: https://github.com/Pika-Labs/Pika-H3-Flex
  inspected revision `4bf8ea508891a432106e70f675d287c27f02306d`.
- Official model repository: https://huggingface.co/Pika-Labs/Pika-H3-Flex
  inspected revision `3ff166447d5edb1406107ab8370c42000ce173e4`;
  `fl2va/manifest.json`, `ref2va/manifest.json` and `comfy/README.md` describe
  the six knots, 12/3 shifts, fresh intermediate noise and adapter conversion.
- Comfy templates: https://github.com/Comfy-Org/workflow_templates
  inspected revision `abb9cf6477b59c52dbe234c67f82318dadd872b2`;
  `video_pika_h3_flex_t2v.json`, `video_pika_h3_flex_fl2va.json`,
  `video_pika_h3_flex_ref2va.json` use LCM, ManualSigmas and BasicGuider.
- User-selected local BF16 adapters: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/loras

The Pika adapter weights are MiniMax H3 derivatives under the MiniMax H3
Community License Agreement, with Mellis, Inc./Pika attribution and notices.
The upstream code license does not replace the model-weight license.
Read the upstream license and NOTICE when downloading or redistributing weights.

Local tests against INT8 H3 and native ComfyUI are not a claim of BF16 bitwise
equivalence, the Pika hosted serving stack, API speed, or Pika endorsement.
