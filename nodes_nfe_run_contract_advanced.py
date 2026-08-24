from __future__ import annotations

from comfy_api.latest import io

from .nfe_run_contract_advanced import compile_nfe_run_contract


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"


class MiniMaxH3NFERunContractT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NFERunContractT8Advanced",
            display_name=(
                "MiniMax H3 NFE Run Contract / "
                "断点运行合约编译器 (Advanced EXP/T8)"
            ),
            description=(
                "Compiles the exact final prompt, media map, conditioning report and real "
                "conditioning tensor contents into strict deterministic JSON for the NFE "
                "checkpoint/resume node. It never mutates conditioning or sampler state."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("positive"),
                io.String.Input("conditioned_prompt", multiline=True),
                io.String.Input("media_map_json", multiline=True),
                io.String.Input("conditioning_report", multiline=True),
                io.Int.Input(
                    "hash_chunk_megabytes",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Upper CPU transfer chunk while hashing conditioning tensors. It only "
                        "changes verification memory/speed, never the contract hash."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("run_contract_json"),
                io.String.Output("contract_sha256"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*compile_nfe_run_contract(**kwargs))


NFE_RUN_CONTRACT_ADVANCED_NODE_CLASSES = [MiniMaxH3NFERunContractT8Advanced]
