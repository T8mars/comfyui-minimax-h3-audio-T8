# v1.97.0 · H07 已审工作流与独立诊断（EXP）

保存[七张通过／接受的原生工作流](../examples/workflows/85-h07-reviewed/README.md)，
旧工作流不覆盖，旧692个节点的完整接口、默认和顺序保留，追加5个入口至697。

- FreeVideo新增可选v0.3.5参考音频t=1加载器，复用主体权重并独立准备新表；
  旧8+2／4+4／四档loader与t=0默认、已有Stage保留，不自动迁移。
- 三个只读诊断：参考编号／实际执行摘要、继续／恢复卡、道具ID／左右手连续性检查。
  不修改条件、自动排队或从MP4伪恢复原生latent。
- 独立交付sidecar：私有本机记录与白名单分享副本分开，默认不写文件／上传。
- 新可见动作半尺寸内部拟音、原画面旁路；[用法](H07_FOLEY_USAGE.md)。
- 专用PDD 7＋learned1.2×＋1：864×480→实际1024×576，原生音画人工通过。
  旧普通Turbo闪光配方不推广，不与需授权的PDD baked混为一谈。
- 背景plate＋原人物配方接受当前有限效果；用户仍认为效果不佳，不宣称完美抠像、
  LanPaint算法修复或已定位模型根因。需自行准备同源MASK及干净前景。
- 清唱采用原录音、匹配Hybrid姿态、同完成Stage Native／X2原配方均保留。
- 原生VAE channels-last-3d权重身份按逻辑字节分块校验；不改权重或采样数学，
  连续布局旧身份不变，坏哈希／未知布局／NaN及producer门不放宽。

七项生成／解码配方经真实画布保存、导出、关闭重开、完整5秒原生音画及人工审核。
此前已通过的四项不重复采样；本次只保存原结果、验证导出和新发行包，不重跑旧CPU矩阵。
公开模板只适配资源入口／说明，数值、seed、sigma、接线与声音路径保持；路径适配
不是新采样质量声明。新三项基础真0.4MP，音频专用半尺寸内部支路明确标注。

## 不在本次资格内

不保证所有输入、声纹、遮挡、风格、语言、模型或高清多窗接缝都通过；TaoMate baked／
PDD baked／SparseRef15尚缺合法访问，指定高清尾／seam材料待补，H07整体不冒称全部完成。
风格LoRA不重复开发，QuantFunc、Nunchaku及暂停的Sol研究不恢复。
模型、原图／视频、配置、Stage、人工反馈和私有roadmap／SKILL不上传或打包。
GitHub版本与Comfy Registry审核／可安装状态独立，不等待或重复上传旧版本。

## 来源与感谢

感谢[FlashML-org/FreeVideo](https://github.com/FlashML-org/FreeVideo)、OpenVDN、MiniMax H3及
相关控制模型作者。FreeVideo SDK与权重／表许可分开，详见[音频时钟说明](FREEVIDEO_AUDIO_CLOCK_EXP.md)。
拟音审片源Jane pouring beer／Angulidayaaluta为CC-BY-SA4.0；归属与修改见[拟音用法](H07_FOLEY_USAGE.md)。
保留并补明确感谢[RK-BoilingPoint/H3-Visual-Marker-Control](https://github.com/RK-BoilingPoint/H3-Visual-Marker-Control)
的原空间标记方法；T8适配不暗示合作或原作者背书。

## English

Seven reviewed native recipes, one opt-in FreeVideo reference-audio t=1 loader, three read-only
diagnostics and a minimal delivery sidecar. Existing692 schemas/defaults/order and old workflows
remain unchanged; five appended nodes make697. New Foley preserves the original image branch
and adopts generated audio only; dedicated PDD7+1 uses the existing learned1.2x upscale.
Background compositing is accepted with explicitly limited quality, not a LanPaint geometry fix.
Gated baked models and a specified HD seam chain remain conditional work, not certified here.
