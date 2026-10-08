# H05：同源中途 Guide（EXP）

独立示例复用 Core `MiniMaxH3AddGuide`，不新增 Guide 算法、不改旧工作流。
Source Clock 同次解码观察 RGB、音频和实际 CFR；Source Conform 只准备一次，
LOW、HIGH 使用同一完整帧映射，各自使用本阶段尺寸。

## 工作流

[H05_Timed_Guide_EXP.json](../examples/workflows/80-h05-timed-guide/H05_Timed_Guide_EXP.json)

先选择自己的合法 5 秒、24fps、有完整音频的视频。固定代表将 120 帧准备为
124 帧，明确记录末帧保持及音频补齐；不承诺任意 VFR 或任意素材的同步资格。
LOW 448×256 → 既有 learned 1.2× → HIGH 实际尺寸，分别采样 4 步。

源第 72 帧对应 3 秒，两阶段分别映射、选帧、由本阶段 VAE 编码后追加为 Guide。
不能把 LOW 采样器直接接到不同尺寸的 HIGH，也不能把两个不同画布的像素 SHA
说成相同。改时间位置需同步两个 ImageFromBatch 与 AddGuide 的帧编号。

Audio1 为完整来源驱动，Audio2 为原片 2–5 秒的独立声样。Guide 追加到既有
条件，不替换声音参考；最终交付使用原片准备后的完整音频，而不是解码人声。
音频补齐、裁齐及 AAC 编码分别报告，不宣称导出 MP4 与原 PCM 逐值相同。
EAV／Prompt Relay 仍可外置显式连接，本示例未默认应用或批准其效果。

## 验证边界

已有 3 项真实 Core 合同测试通过，覆盖非零／负帧编号、保留 RefAudio、
越界编码前拒绝；测试 encoder 不是预训练权重，不代表 GPU 画质。
实际同源帧映射 CPU 预检通过。新工作流已经真实画布另存、重开、原生 Run
一次：47 执行节点／88 条边，69.229 秒，LOW4＋HIGH4、learned1.2×，完整
120 帧／512×288／24fps／5 秒的图像、音频和 PTS 已检查，浏览器播放至结尾
有画面。两个实际 Guide PNG 与对应来源帧的阶段 resize 像素逐值一致，
LOW／HIGH 的实际帧映射 SHA 与原同源预检一致。

原生 LoadVideo 请求附加一个空 `video-preview` UI 字段，保留原始比较失败；
真实 Core 输入函数独立确认只消费 `file`，不是改图或忽略未知采样参数。
声音参考实际 ordinal 与原驱动职责保持；未另外捕获原生编码参考 tensor，
不把 CPU 保留对象测试冒称实际声纹批准。音频锁审计在原容差内重新放回输入
audio latent；最终 FLAC 预览相对原 MP4 解码 PCM 最大差为 1/65536，
是编码量化，不写 exact。MP4 AAC 也不宣称原 PCM exact。

仍待最后人工审核，不把原 N01 审核借给新 Guide 效果。

未复现声音参考丢失反例，不升级或回补共享 Core。不保证 Guide 是逐像素硬锁，
也不保证任意角色、任意时间位置、多人演唱或多窗均已通过。

## 创建自己的独立图

```powershell
python tools/prepare_h05_timed_guide_workflow.py --server http://127.0.0.1:8189 --output H05_Timed_Guide_EXP.json
```

工具只读取本机实际节点 schema，并 create-only 创建文件，不安装、下载或提交任务。
目标存在会拒绝覆盖。模板需要已有 T8 模型、Qwen、原生视频／音频 VAE、
4 步 adapter、learned 3D upscaler 与 KJNodes。
