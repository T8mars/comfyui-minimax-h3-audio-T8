# RES 完成态保存／读取（EXP）

两个追加入口不改变原节点 schema、Core RES 数学、历史检查点格式或旧工作流：

- `MiniMaxH3RESStageSamplerEXPT8`：保留原生五个采样输入和两个 LATENT 输出，
  另输出 typed StageResult、StageContext 和报告，接现有 Stage Save。
- `MiniMaxH3RESStageLoadEXPT8`：指定已完成 manifest 的相对路径、必填 SHA 和
  `res_complete`／`res_remaining_complete`，只读恢复两个 LATENT；不加载 MODEL／
  CLIP，不生成 noise，不采样，也不是多步历史续跑。

## 两种保存不能混淆

RES History 保存第4步更新后的 x、old_denoised 和 old_sigma 等完整求解历史，
用同一原始 MODEL／条件／noise／latent／mask／完整 sigma 轨迹恢复剩余步数。
完成态 Stage 保存的是完整原生 sampler 已返回的两个最终音画输出，不是
partial x0，也不把同轨历史边界转成 LOW→放大→HIGH 的接力状态。

Stage 只接受实际 RES History 工厂和未改的 KSAMPLER 选项。它观察真实求解器
每次模型返回、全局／局部 callback 和完成返回，保持执行 MODEL 及 wrapper 类型。
未知 MODEL 补丁仍可执行，报告未认证；不以未知对象建立可移植完成态复用身份。
现有 Stage Save／Load 的数据-only safetensors、显式 path／SHA、原子 manifest、
内容检查和只新建不覆盖规则不变，文件位于 `output/MiniMaxH3/stage_artifacts/`。

读取明确冻结所选旧结果，不承诺今天修改后的参数仍匹配该结果，不开启自动缓存。
只有实际已认证计算内容、条件／位置、noise 和编译资产与完整结束证据都满足时，
才允许这种明确选定的完成态复用；不翻转通用 MODEL 可移植或 CUDA 数值资格。

## 实际验证范围

七项直接相关 CPU 行为检查和一项真实入口注册检查通过，分属已保留源码阶段。
包括完整8步／恢复剩余4步的真实 tiny Core 数值、两输出保存读取、原历史节点
双向兼容、未知 MODEL 实际运行但不认证、异常不产生完成结果、伪造完成计数／
输出篡改与错 sigma 拒绝。没有复跑旧 N01／solver，也不称当前全仓测试通过。

实际固定 Ref2VA INT8 ConvRot／FFN2／KJ SM89／原版 Sol 画布，保持原 seed、
条件及第4步边界，只执行剩余4步。新 Stage 图原生另存、Save、关闭标签及从
侧栏重开后 Run，24节点／31连线，170.349秒，无节点执行缓存；两个完整输出
另存原生 AV 数据并保存 typed Stage。完整120帧、512×288、24fps、5秒 RGB／
解码 PCM 及四个最终 float32 AV tensor 与原连续 RES 逐值一致，原边界 SHA 未改。

独立读取图同样原生另存／保存／关闭重开／Run，10节点／11连线，15.848秒。
没有 MODEL／CLIP／noise／guider／sampler／历史节点，新增扩散 NFE 为0；仅两个
VAE 加载节点命中缓存，Stage Load 与 Decode 确实执行。两个读取输出再次保存，
所有持久字段及完整成片仍与原结果一致。浏览器真实播放到5秒结尾，有可见画面、
readyState=4、无媒体错误。FLAC 预览分别持久另存，AAC padding 不与 FLAC 混称相等。

原 T8 live schema 全字段及顺序保持；VHS 图像扩展注解在两个进程中的列表顺序
不同，原差异单列保留，不称整个第三方 schema 逐字段完全相同。

当前只批准这份固定计算链的明确完成态复用，不保证其他 GPU、任意 hook、
所有完整8步图或其他几何。外置 EAV／Relay 新组合及人声／口型／画质人工审核
仍待完成；未获人审的验收图留在本机，不作为推荐公开示例或已发布功能。
