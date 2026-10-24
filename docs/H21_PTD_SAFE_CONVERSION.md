# PTD DCP 安全转换（独立 CPU 工具）

`tools/convert_h21_ptd_safe.py` 默认只准备，不转换、不导入 Torch、不读取 shard 张量、不启动服务或 GPU。它不调用 `pickle.load/loads`、Unpickler、`FileSystemReader.read_metadata`，没有 `--trust-dcp`。

## 默认准备

```powershell
python -I tools/convert_h21_ptd_safe.py --source <DCP目录> --recipe h3_4step --report <新的报告.json>
```

`h3_t8/ptd_safe_metadata.py` 使用 `pickletools` 解码字节，但以自己的惰性数据记录解释闭合协议，不导入 checkpoint 命名的模块，不执行 `NEWOBJ/BUILD/REDUCE` 指定的 Python 函数。仅支持显式协议 4/5 的张量型 DCP 数据结构。未知符号、持久对象、BYTE_IO、storage transform、坏 memo、尾随程序和不支持的 opcode 均报具体格式错误，不以“可信下载”代替安全读取。

每个 tensor、chunk、storage 都检查实际类型、维度、偏移、覆盖、重叠、规范相对路径与真实文件边界。分布式写入的合法空尾 shard 保留并检查，不误作坏矩阵。原始元数据可能扩展为许多 Python 记录，读取本身需要真实 CPU 内存。

只明确排除发布格式的完整曲率训练头列表，并写入报告；不静默丢弃未知条目。A/B 配对、hop、完整拓扑及实际维度检查不取代 MODEL 运行时目标维度检查。不同 hop／独立 QKV 的不同秩保留并警告，不为方便堆叠而强制同秩。源声明未知时输出保持 `Unknown`，不伪装固定底模来源。

## 显式转换

```powershell
python -I tools/convert_h21_ptd_safe.py --source <DCP目录> --recipe h3_4step --convert --manifest <固定文件清单.json> --manifest-root <清单相对路径根> --output <源目录以外的新文件.safetensors> --report <新的报告.json>
```

清单格式为 `files: [{file, bytes, sha256}, ...]`，必须覆盖元数据和全部引用 shard。转换前后重新核实际完整 SHA、文件大小和清单字节；不是只看 mtime。清单所有目标必须位于明确选定的 checkpoint 目录中，不跟随 metadata 去读其它目录。

每次仅把一个有边界的序列化 storage slice 交给真实 `torch.load(..., map_location="cpu", weights_only=True)`。当前 Torch 冷导入会自行登记其 `NestedTensor`／`_rebuild_njt` 及 DeviceMesh 的数据类 `_MeshLayout`；仅接受这三个实际 Torch 对象的身份，不按名字接受替身或别名。用户／Core 添加的其它 `safe_globals` 仍拒绝，因为这会扩张可执行 callback 的信任边界。工具不替用户清空或修改 allowlist。任意对象、张量子类、错 dtype／shape、NaN/Inf 及非 CPU 稠密张量都不能写成有效结果。

输出按原逻辑元素逐 chunk 写入 safetensors，没有完整专家库张量分配、A/B 合并、重采样、量化、转 dtype 或 SwiGLU 行序变换；原运行时继续负责映射。单 chunk 可能需要 contiguous 副本，操作系统文件页缓存也有真实成本，不承诺零复制或固定内存最低值。仅在小端 CPU 写入原字节。

`weights_only` 不是 DoS 防护或内存配额：恶意 tensor 可以声明比其序列化 record 更大的 storage，文件字节数本身不限制解包分配。这里的逐 chunk 内存预期仅针对已经核对 SHA 的固定来源。使用独立拥有、转换时不再变更的源目录；前后 SHA 扫描也不能替代操作系统对蓄意并发写入者的隔离。

临时输出在目标目录创建，验证 header 后以不覆盖的原子 hard link 发布；已有用户文件不替换。出错删除本工具尚未发布的临时文件，不改原始 DCP。新的 safetensors 自身带来源 metadata SHA、准确训练排除项及格式标签。完成转换仍不表示模型组合、GPU 速度、画质、音频或人审已通过。

## 已核对的真实来源与边界

固定 PTD 权重 revision [`8a70e31c`](https://huggingface.co/epfl-vita/PTD-weights/tree/8a70e31c82a7ce6520dbfa334ebbe633780f85ed) 的 4 步目录包含 30,495,526 字节 metadata、2,532 个张量、161,428 个 chunk；11 个实际符号均在闭合数据结构内。默认完整准备识别 50 主块／2 refiner／1,256 对专家，明确排除 16 个训练曲率头，推理张量 11,082,695,168 字节。一次真实准备约 8.56 秒、进程峰工作集 354,480,128 字节；不是转换峰值或机器最低配置。

源格式依据固定 [PTD-inference `43ae22c3`](https://github.com/vita-epfl/PTD-inference/tree/43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85)。MiniMax/Hugging Face 原组件、EPFL VITA Lab 的 PTD 贡献与相应 Apache-2.0 NOTICE 归属保留；权重与输出仍受其 MiniMax H3 Community License 约束。此工具不替用户接受许可证，也不分发已下载权重。

截至 2026-10-22，4／6 步两份整包安全转换已完成；独立核对全部 6,288 个 tensor、401,008 个源 segment 及合法空尾记录，未用小型合成张量代替完整来源检查。转换输出分别为 11,083,041,896 和 16,623,188,376 字节，完整 SHA 与独立源字节收据记录在开发目录 `ptd-nvme-conversion-qualified-v1.json`。

4 步训练模型已实际完成 T2VA 单采／连续 2+2 的 raw AV 和完整媒体对照；6 步完整采样／连续 3+3 也已独立实测，最终 raw AV 逐字节相同，两片完整媒体机械通过。该范围不包含冷恢复、放大、原生 40 步或其它条件模式；转换和选定采样资格都不代表性能或人工画质通过，具体边界见 [PTD 节点说明](H21_PTD_EXP.md)。
