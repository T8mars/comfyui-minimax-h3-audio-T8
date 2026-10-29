# Open-H3-Regenerate-2K integration references

The T8 integration is independently written for the existing ComfyUI H3
runtime. It delegates native sampling and linear LoRA application, and adds
scoped Sharp curve evaluation. It does not redistribute the author's models,
input videos, Sol-Engine implementation or fast VAE engine.

- [Author project and code](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K): adapter composition, independent Sharp curves, sigma schedules, input preprocessing and learned latent upscale references. The inspected raw branch files are bound by content SHA below; this is not a verified Git commit pin.
- [Author model repository](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/tree/5809c33730c18b6b1c827b40974855a4288641b3), selected revision `5809c33730c18b6b1c827b40974855a4288641b3`. Its model card identifies the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE); model assets are separate from this node package.
- [Kijai experimental models](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/loras): the separately supplied rank384 BF16 pure-refiner extraction. It omits Turbo and Sharp and is not described as the complete recipe.
- The author's merged linear factors include LightX2V FL2VA Turbo v1.2 and LMS Sharp. Their strengths are already folded into the factors; T8 does not apply the Turbo alpha/rank twice. See the author's acknowledgements for each component's source.
- Learned 3D latent upscale uses the existing T8 implementation and selected [H3 latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) model. HyperFlow uses the existing dedicated loader, not a generic LoRA substitution.

Selected source content SHA256:

| Author path | SHA256 |
| --- | --- |
| `open_h3/adapters.py` | `3701ab90e4c6c0c066ab7bcd8fb3cd5f4d23578fac1c27fe6ac0f35f9b239db2` |
| `open_h3/standard_fwd.py` | `8291b35f8249a2b37c1ce1b0146bd87b08dcdd73c0d8a9e17259e30b2234f40e` |
| `open_h3/sigmas.py` | `6c614edff70aa7f53699badc24fa2d6af0604fb6115ccd7825b3547affdd8c51` |
| `README.md` | `bc7290656203f1ae54ddd92460348504dbb3ebbdc3dc849af5e570d14ccce8ad` |

Selected merged weight SHA256:
`8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04`.
Selected independent Sharp curve file SHA256:
`26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e`.
Selected 1025×8 Ref2VA `adaln_t_table` tensor SHA256:
`c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7`.

These identify the validation assets; they are not a model allowlist. Local
INT8/NVFP4 Core execution does not establish the author's BF16 bitwise results,
Sol-Engine performance or human acceptance of generated media. Validation
scopes and saved workflows are described in [the integration guide](../docs/OPEN_REGENERATE_2K.md).
