# v1.88.0 — Veda 与独立采样阶段（EXP）

本版交付Veda和S01–S29分离式采样入口。现有一体节点、保存工作流、原步数／sigma／AV时钟、学习型3D放大与已接受接缝配方保留。新图为opt-in，不强制迁移。版本号不等于Registry已Active或人工画质通过。

## 使用入口

- [全部分离采样](MODULAR_SAMPLING_EXP.md)：按配方选择不同类型的阶段、交接和保存／恢复节点，不把中间x_sigma、预测x0或完成AV混接。
- [Veda T2VA](VEDA_SPARSE_T2VA_EXP.md)和[8步工作流](../examples/workflows/63-veda-t2va/README.md)：官方新文件是tile-score预测器，不是新的H3底模；直接由Bundle读取，无需转换成UNET。H3底模、Turbo v4 step600 LoRA和原生8步另接。
- [多人独立来源保存](MULTIFACE_FROZEN_SOURCE_EXP.md)：每个角色独立保存完成Stage及当次plan／窗口／AV；冷图不重跑SAM、编码器或H3采样。两个存档各有自己的路径和SHA。
- [LTX原生加载策略](LTX_NATIVE_LOAD_POLICY_EXP.md)、[动漫隔离交付](FACE_ANIME_ISOLATED_DELIVERY_EXP.md)：只在对应新图显式启用，旧行为保留。

每个阶段可以独立选择MODEL、LoRA、提示条件和支持的外置EAV／Prompt Relay。配置存在不代表实际执行；使用对应Audit核对真实覆盖。未知用户补丁保留／委托并说明资格，不静默删除或强制禁止。

## 兼容与安全

旧575节点到本地581节点的审查逐项限定注册、资产菜单和实现变化；原2098份JSON字节不变。严格旧M0原始报告仍记为需要审查，不改失败证据或泛化放宽schema。新增12份图与旧图分开。

显式保存使用数据型safetensors／manifest，绑定真实来源、阶段、文件SHA及原音轨，不包含模型或可执行对象。错误阶段、混角色、坏哈希、越界路径和partial文件明确拒绝，不偷偷重采。未知执行组合不会凭文件名获得跨进程缓存认证。

未来新采样入口有AST／导演台构图及分离图准入检查，需要独立阶段、外置效果和恢复接线；它不是任意未来模型的自动画质认证。发行布局冻结86个实际调用点；原105项研究清单保留，其中19项来自未发布的根目录旧副本，实际h3_t8调用点及完整导演台构图指纹不变。

## 实证及限制

真实完整原生画布／新进程验证包括双人、三人原8步、动漫原12步、局部及手动窗原8步、独立LTX原3步等代表；具体证据边界写在各专题文档。多人完整124帧画音和冷侧0采样逐值相同，四份新多人图也通过实际另存、刷新／重开和严格序列化检查。

Veda动态编译修正后的匹配重复结果一致，但首次编译包含的配对速度门没有通过；只记录指定暖运行优势，不保证首次或普遍提速。Relay逐query路线明确记录Dense委托，不把它当Sparse执行。用户取消额外十二网格／十八素材矩阵，不追加运行，也不借未运行网格声明完成。

完整当前CPU回归覆盖271个完整文件、4416项，另38个导演台完整文件429项通过（含真实服务界面／既有成片恢复），均无跳过或排除。后续元数据索引与发行布局单独精确复验，不将重叠计数相加。发布门仍要求官方CLI实际包、真实Core注册／旧schema及线上294份原JSON逐字节检查；GitHub上传不等于Registry可安装。机器功能与人工画质分别记录，新成片主观审核最后由用户确认。QuantFunc／Nunchaku不恢复，模型／私有roadmap／验收视频不打包，自动Registry监控不重开。

English: opt-in Veda and independently wired sampling stages preserve legacy graphs and numerical contracts. Explicit per-stage effects and source/stage data storage are audited separately from subjective quality. Veda is a predictor paired with the original eight-step recipe, not a new UNET; measured warm benefits do not guarantee first-use speed. Current complete CPU regression passed4416 cases, with429 additional Director checks including live UI; overlapping scopes are not summed. Actual distribution and compatibility remain release gates. Registry availability and human quality acceptance remain independent.
