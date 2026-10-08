# H05 MV：同一原点、三路音频（EXP）

新增独立模板，不修改原 MV、普通双采或旧音频默认。它复用现有节点，
将 **驱动人声、声线参考、最终混音** 分开，避免用声样当目标时间线，或
把一采尚未完成的原生音频冻结成最终声音。

| 音路 | 用途 | 示例接线 |
| --- | --- | --- |
| DRIVE | 决定目标片段的表演时间 | 完整人声 stem → 共享原点裁切 → LOW/HIGH `drive_audio`，Audio1 |
| VOICE | 只提供声音特征，不代替歌词/时间线 | 独立声样片段 → LOW/HIGH `ref_audios`，Audio2 |
| FINAL | 保留原混音作为交付声音，不参与生成条件编码 | 完整混音 → 同一原点裁切 → `final_audio` → Output Trim → Safe AV Save |

人声 stem 和混音必须来自同一 master，且在原文件中已经时间对齐。
示例共享原点为 5 秒；124/24 秒准备范围在最终交付时裁成 120 帧、5 秒。
声样独立从 16 秒取 3 秒，不代表目标对白/歌词从 16 秒开始。
可人工改原点、提示词、歌词、拍点与角色；没有自动转录、自动 diarization、
自动重新分配角色或自动覆盖歌词功能。没有源音频的 Source AV silence
不能称“保留原声”；需要重采样、补齐或裁齐时必须另记处理事实。

## 工作流

[H05_MV_Master_EXP.json](../examples/workflows/79-h05-mv-master/H05_MV_Master_EXP.json)
使用占位素材名，先选择自己的表演者图、完整人声 stem 和原混音。
需要已有 T8 模型、视频/音频 VAE、Qwen、4步 adapter、learned 3D upscaler
以及已安装的 KJNodes；不自动下载或安装。

LOW4 → 原 learned 1.2× → fresh HIGH4，各自使用按本阶段实际尺寸建立的
Dual Clock Sampler。HIGH 不可直接复用捕获 LOW packed geometry 的 sampler。
同一 sigma 计划拆为 4+4，不改变旧采样器算法或尺寸守卫。

本模板显式选 `lock_source` / strength 0，并由 HIGH template 保留已经完整的
外部驱动音频。这不等于允许把部分 LOW 原生生成的音频锁住。音频审计会检查
真正的 mask、有限值和误差；已有审计节点会在容差内重新放回输入 audio latent。
最终 mux 始终选择 **原混音 AUDIO**，不是这个 latent 的 VAE 解码声音。
EAV、Prompt Relay 保持可外置接入，但本示例没有默认应用或宣称其效果通过。

## 已取得的验证及边界

一条修正后真实画布 SaveAs → 重开 → Run 的 4+4 / 1.2× 代表已完成，
39 个执行节点、65 条边和实际执行参数完全一致；58.830 秒，120 帧、
512×288、24 fps、精确 5 秒。全部视频帧/PTS 与立体声 32k 音频已解码检查，
浏览器完整播放到结尾且有画面。实际 LOW/HIGH 都是 Audio1 驱动 / Audio2 声样。
源音频和实际 DRIVE/FINAL FLAC 预览的对应切片逐值一致；MP4 AAC 有编码及尾部
补样，不宣称原 PCM exact。独立 CPU 合同用例覆盖非零原点、三路职责及缺音频。

代表复用了既有 **spoken-dialogue MV** 的完整人声与混音，不把文件名推断成
歌曲，也不冒称新演唱/歌词/多人歌声质量已验证。最终画面、口型、声音质量
仍待集中人工审核，不作为自动质量通过或推荐配方。
首个新图的 LOW sampler 误用于 HIGH 所导致的失败已保留；只修新图两条连接并
追加 HIGH sampler，没有放宽守卫。没有给旧工作流增加节点或改变默认。

## 显式创建自己的图

```powershell
python tools/prepare_h05_mv_master_workflow.py --server http://127.0.0.1:8189 --output H05_MV_Master_EXP.json
```

工具仅查询指定本机的实际节点 schema 并创建新图；目标已存在则拒绝覆盖，
不提交 Queue。生成后需在真实画布另存、重开、核对所选素材并验收。
模板/工具本地可用不等于已发布 GitHub/Registry。
