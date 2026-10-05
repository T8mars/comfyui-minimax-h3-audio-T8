# RADAR r6 精确尺寸与原片救援（指定裁切对照已人审）

这是N05的最小交付合同，不修改采样器、latent、MASK／Face回贴或旧保存节点。内部模型尺寸仍遵守32格点，最终完成的RGB才裁切。

## 精确裁切

本机Core使用 `ImageCropV2` 的 `crop_region` BoundingBox控件，不是旧版width／height四个API端口。当前schema声明 `socketless=true`，画布不能接 `PrimitiveBoundingBox`；在节点的裁切框里明确设置x／y／width／height。传统画布会把框展开成四个控件，实际API仍是一个BoundingBox字典。例如完成1920×1088 RGB之后，x=0、y=4、width=1920、height=1080。必须放在最后一次MASK／Face回贴之后。CPU能用PrimitiveBoundingBox构造字典，不代表画布有该插口。

中心裁切会删除上下内容，不是免费保留所有画面。用户不允许裁内容时应明确选pad或resize另配方，不偷偷换策略。不得把非24fps外片直接接固定24fps保存而声称时钟未改；先核实际帧率、PTS与音轨。

## 原片先保存

1. 完整双采及所有回贴完成后，用已有安全保存节点持久保存master；该节点严格解码、原子新文件发布并返回VIDEO／路径／输出SHA，不覆盖已有文件。
2. 后处理以这个已保存master作为实际依赖，不能把独立Save根“放在画布左边”当执行顺序保证。独立新文件失败时保留master及SHA，报告后处理失败，不展示增强成功。
3. 像素后处理的声音来自这个master的音轨，逐包复制，不再编码AAC，也不再生成声音。原始AUDIO只用于第一次安全保存；后处理不拿解码后的AAC重新编码冒充原始PCM不变。视频与音频分别核PTS，并复核解码PCM。

已核现有Core的2帧1920×1088→1920×1080精确RGB裁切，并用1秒小型CPU测试信号检查同次队列：原AUDIO不变、两份实际AAC解码PCM和音视频PTS相同；注入下游编码失败后master SHA不变、失败目标未发布。前两次测试夹具分别因Core nodes路径被遮蔽、comfy为namespace包而失败，证据保留；显式从真实Core工具模块定位后2项通过，不改生产或减弱断言。测试信号不是模型成片或人审样片，也不证明所有音频编码器均逐位确定。

## 可打开模板与失败出口

三个短名模板在 [74-radar-r6-delivery](../examples/workflows/74-radar-r6-delivery/README.md)：Full master裁切附加模块、Cold master裁切、Cold master等比例缩放加留边。Full输入是全部回贴完成的RGB和AUDIO，不是另一个采样工作流；Cold没有模型／采样／Stage依赖。Cold的文件占位必须换成自己的完整master；缺文件警告不是可用缓存，不能盲目排队。

新增一个薄封装 `MiniMaxH3PostprocessSaveEXPT8`：master VIDEO、完整后处理IMAGE，默认确认false；沿用既有isolated encoder、音频包mux、媒体校验与原子不覆盖publisher。旧Skin Finalize只允许原几何，不能替代这个裁切出口；这里没有第二套publisher或生成引擎。旧656节点的完整schema、默认值和顺序保持，该节点只追加。

- 成功：`postprocessed_video` 是新文件，`master_video` 是独立原片出口；状态 `postprocess_complete`。写 `.mp4.postprocess.json` 成片回执与 `.postprocess-state.json` 作业状态。
- 失败：状态 `postprocess_failed`，新成片出口是ExecutionBlocker而不是把master伪装为增强片；原片仍在独立出口和回执路径。保留原片SHA、失败原因，不显示成功播放器。
- 已建立作业之后取消：写 `postprocess_cancelled` 并重新抛出中断，不吞Comfy取消。未确认时不写后处理文件。
- 冷读状态必须同时核实际master、实际成片字节和侧车绑定。running或孤立receipt不等于完成；同目标不能覆盖另一个作业。

限定完整零起点24fps SDR，时间长度／帧数不变。VFR、HDR、非零视频起点、trim或临时内存VIDEO不能偷偷转换成这个合同。宽高需要为偶数。全帧IMAGE仍有RAM成本，本节点不是长视频流式内存优化。H264重编码改变画面及容器字节，不能用输出MP4 SHA等于master证明保音；声音依赖包／PCM／时间线独立对照。

## 当前验证边界

27项受影响CPU测试通过（模板真实Core schema、冷裁切、旧publisher、原片保存、失败／取消／错SHA／裁时间和机制对照），0skip／CUDA未初始化，冻结4979源无变化。一次真实原生画布复用已通过的5秒成片：LoadVideo→GetVideoComponents→Core裁切框→后处理保存，512×288→512×280/y4，120帧／24fps，0新采样。成功后独立CPU新进程严格解码并核音频包内容／时间线、PCM／时间线均exact、原片SHA保持，编码前RGB SHA与该master实际裁切一致；不是无损H264或新生成人审认证。

三模板均已实际原生另存、关闭画布后从工作流栏重开；Full与Contain未排队它们缺输入的占位。Cold的原生另存保持实际参数及接线，新节点临时诊断文本不写入保存字段。指定实际master／裁切片对照已获用户通过；Full可组合性和Contain接线仍是CPU／画布资格，不借该裁切片证明任意后处理、所有素材或质量。
