# S19–S21 Chunked v2/v3/v4 分离式双采（EXP）

本目录按 v2／v3／v4 × 10 种外置效果配置 × 完整不保存／完整保存／冷 HIGH 三种形式提供 90 张可导入图。十种配置是无效果，或 EAV、Prompt Relay、两者组合分别作用 LOW、HIGH、两阶段。旧 Chunked 一体工作流和原采样节点没有替换。

建议先导入 `none_none_full_no_save` 看原始两阶段，再导入 `combined_both_full_save`：LOW 和 HIGH 各有独立 MODEL、条件和采样接线，Relay Plan 与 EAV Config 可按阶段分别编辑。v2 保留 PDD 绝对 LOW4／HIGH4、全片单块与原全局噪声同种子；v3 保留完整原生 LOW8／低 sigma HIGH3；v4 在 v3 基础上保留原视频遮罩继承。两阶段的默认权重可以相同，但 HIGH 有单独的模型加载节点；改变配方、尺寸、时间块、遮罩或噪声须重新验证，不能把这 90 张图当作任意 Chunked 参数的通用模板。

`full_save` 的 LOW 和 HIGH 原生 AV 保存都默认 `confirm_save=false`。需要冷恢复时，先开启 LOW 保存并记录返回的真实相对路径、完整外部 manifest 和文件 SHA256，再填到对应的 `cold_high` 图；冷图没有 LOW 模型／采样／效果节点，只从已冻结 LOW 继续 HIGH。文件名中的 `low`／`both` 表示完整图里 LOW 的效果选择；它在冷图里是已冻结的历史，不会重新运行或根据当前控件自动认定等价。改了 LOW 模型、素材、提示词或效果，就重新运行 LOW 并保存。坏 SHA 会在 HIGH 采样前拒绝。HIGH 保存可供另行审计，但这里没有“已完成 HIGH 只读取”图。

先替换首／尾帧、v4 遮罩、冷恢复回执占位，再核对本机 FL2VA、PDD 或 Turbo LoRA、Qwen、双 VAE 和 learned3D 权重。EAV 默认 `report_only`，不施加增益；`apply_exp` 的画质需另行验收。Relay 与 EAV 的组合仍按各阶段实际调用审计；v2/v3/v4 的 HIGH 专属 Chunked 绑定不应直接换成普通 Stage 绑定。末端使用 Core 原生 24fps 视频封装，旧 VHS 图继续保留；不声明新旧 MP4 容器字节相同。

当前 90/90 图通过当前 Core 静态校验、真实隔离 Chromium 原生另存重开及语义审计，六张组合图实际编辑 Relay/EAV 后保持。三条路线各有小尺寸 tiny Core 的新图完整／保存 LOW／冷 HIGH 最终 AV 相等与坏 SHA 拒绝；LOW Relay→EAV 顺序另经 tiny 实采。此前 SHA 固定私有图的缩画布真实权重 GPU 画音机械证据**不自动转移**到这些新通用图。原尺寸、多空间 tile、更多资产／后端、任意设置旧新逐帧画音、EAV 画质和人工接缝／声音验收都还未完成，不要把可导入性当成可直接排队的质量推荐。
