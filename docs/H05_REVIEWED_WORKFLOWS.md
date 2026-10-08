# H05 通过工作流

打开[83-h05-reviewed](../examples/workflows/83-h05-reviewed/README.md)。11项是
2026-10-08人工审核的指定案例，不是任意模型／时长／设备／素材保证。
原生保存图和素材保留本机；公开模板不附本机路径、缓存、模型或素材。

| 编号 | 工作流 | 注意 |
| --- | --- | --- |
| A1 | CupHand Composite | 仅回贴，无采样/VAE。同源原片、已完成RAW和120帧MASK；blend5、音频替换0 |
| A2 | PixelTune50 | 原件stable v3／strength1／原生50步、FL2VA pruned INT8 ConvRot；非作者BF16数值复现 |
| B1 | RES Checkpoint | 完整8步保存第4步求解历史；恢复是同轨余4步，不是LOW/HIGH放大 |
| B2 | RES Relay | 外置两事件Relay apply、EAV report_only；报告模式不叫增强画质 |
| B3 | RES EAV Low | 无Relay；tau1.5／视频进度10%–90%／上限1.5，实际gain全1 |
| B4 | TwoVoice 12s | 双角色参考／全局对白；二采187／overlap34，两HIGH窗各4步；非硬声纹隔离 |
| B5 | Master Audio | 同原点DRIVE／VOICE／FINAL、原录音末端交付；本次是对白，非歌唱／歌词验收 |
| B6 | Timed Guide | frame72=3s、LOW/HIGH共用帧索引、learned1.2倍；非任意姿态锁定 |
| B7 | LTX Source | 外片CFR30→24、256×448／LTX三步／H3零步；一事件Relay及EAV报告模式 |
| C1 | FV Decode Pair | 同一已完成Quality HIGH／FP32视频VAE eager与compile；0新增采样 |
| C2 | Dense 4+4 | LOW448×256→HIGH512×288／原learned权重／1.2倍，非完整Flow或异构接缝保证 |

## 输入和依赖

加载自己的合法素材和对应模型。占位文件不会自动下载；KJ Sage、原版Sol、
VideoHelperSuite、LTX节点按各图单独安装，不改变所选后端。

A1的生成MASK与回贴MASK不同。固定通过案例仅完整可见F0–64／F100–119扩大12px；
部分遮挡F65–73／F99不扩、完全遮挡F74–98保持0。不是其他素材可盲用的时间表。
先核同源和遮挡。RAW来自[独立LanPaint时钟入口](H05_LANPAINT_CLOCK_EXP.md)，
外部GPL-3.0采样器独立安装，新适配实际源码pin不匹配时报错。旧Prompt First／
blend1失败回贴不作推荐。

B1/B2/B3历史保存create-only，每次新任务改短checkpoint名。恢复要保留原始MODEL、
LoRA顺序、seed、条件、AV输入、MASK、sigma和效果配置，并使用同一文件。
图中人类标签／旧资格摘要不代替实际内容检查。未知补丁可运行，不被认证为可移植缓存。
详见[历史](H05_RES_HISTORY_EXP.md)、[完成态](H05_RES_STAGE_EXP.md)、[外置效果](H05_RES_EFFECTS_EXP.md)。

C1先用真实本机配置完成FreeVideo HIGH，再填Stage路径和实际manifest SHA；
按[独立解码说明](FREEVIDEO_DECODER_EXP.md)生成decoder配置。空SHA故意阻止误用
别人的缓存；不关闭检查。compile显式选择，不自动回退或保证通用加速。

生成音频、源音轨锁定、末端AAC不同，不把AAC padding叫PCM exact。
其他H05许可／特定来源／硬件研究未因这11项通过而全部完成。
