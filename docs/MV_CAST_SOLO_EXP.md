# A/B 交替独唱 MV（本地 EXP）

通用 Full／Same Output Resume 图在 [69-radar-native-recipes](../examples/workflows/69-radar-native-recipes/README.md)。两张图执行参数完全相同，仅说明不同：首次选择自己的参考／对齐音轨／完整场景分配并用新chain，恢复保持同目录及原请求，不能当作免producer加载的任意移址解码图。

新增两个独立节点，不改变旧 MV V1–V3 的控件、默认值或已保存工作流。CPU合同、真实两镜画布完整成片及同一保存output目录的新进程零采样恢复已通过机械检查；人物身份、口型和整片看听仍待人审，未发布。

使用原 VocalLock Scene Planner 得到 scene_plan，再接 `MiniMaxH3MVCastSoloPlanEXPT8`。每场景明确一个零基 `scene_index`、`performer_id`（A或B）和 `exact_vocal_text`；必须覆盖全部场景并实际使用两名角色。空歌词表示未提供原文，不猜测ASR、说话人或歌词。提供的文字和空白保持原样，但不能含原生控制标签。

Renderer的 `reference_a` / `reference_b` 各接恰好一帧IMAGE。二者可不同尺寸，不自动堆叠、裁切或猜角色。每镜独立使用选定角色的Ref2VA参考，first/last为空，不借前一角色尾帧延续身份。这是A独唱→B独唱，不是同时同框合唱；多人同框另需实现，不作承诺。

full_song与vocal_lock_audio沿用旧VocalLock时间对齐检查和原24fps场景坐标。条件音轨按原坐标截取；最终完整原歌只mux一次，不多次拼接重新生成的声音。内部accepted是文件/合同机械组装状态，**不是人审通过**，报告固定 `human_quality_accepted=false`。

新路线恢复绑定完整两份参考、双音轨、实际选定MODEL/CLIP/视频与音频VAE，以及实现内容；不只用旧64值变化检测，也不相信model_id标签。任一参考、歌词cue、角色、音轨或已识别producer变化需新chain；渲染期间变更在保存/机械接纳前拒绝。未知用户wrapper保留执行但nonce-bound/nonportable，不能借名字获得跨调用资格。

最小CPU检查覆盖实际A/B对象选择、无前尾、一次mux、恢复、不在64值抽样位置的图/声音变化、实际producer变化，以及重签但内部矛盾的plan拒绝。Mock采样不认证真实模型、人物身份、口型或成片；使用专用Ref2V Turbo4 LoRA的默认4步也仍需独立真实资格，不自动下载/改模型或套到其它Turbo配方。

实际首轮画布在第一镜4步及解码后被新producer核对拒绝，原因是Core采样clone共享root网络并留下时钟对象；原失败完整保留，没有接纳该镜。新cast路线单独使用原生clone的root module表及对象补丁backup容器，扩散权重、LoRA、hooks和实际加载语义保留，继续委托原单镜采样器。六完整CPU范围168项通过，新增真实Core tiny AV无LoRA/非零LoRA对照：原clock不变、4步画音输出与旧采样逐值相同、重复一致、声明式权重/LoRA内容变更仍拒绝。旧584完整schema/2029本地JSON仍保持，CUDA未初始化；不能将CPU结果当成真实A/B人物成片或人审通过。

后续完整模型仍在第一镜后的校验失败；只读字段定位确认另一项变化是已有H3音频编码shim把Audio VAE.crop_input从默认true改为false。新cast路线现在在绑定前调用这条既有、幂等且只针对H3音频VAE的规则，实际编码政策不变。字段仍参与合同，晚改回true会在保存/接纳前拒绝；不通过忽略crop或重签绑定解决。当前七完整范围175 CPU通过，两项原库warning，无skip/deselect、CUDA未初始化；三个原生失败均保留，完整两镜和fresh验收仍待。

上述是修复时的历史截面，不覆盖后续资格：实际画布 Save As/重开/Queue 以独立参考 A/B 各原4步完成248帧448×256/24fps（10⅓秒）H.264/AAC，完整原歌仅末尾mux。后续在同一保存output根的新进程真实 Save As/重开/Queue 返回已核对master，零采样，完整解码 RGB/PCM 与full逐值相同。源、五份实际权重、控制器、原输入和原保存output全部字节稳定，自有服务已关闭。

最初复制output到另一个目录的尝试仍因旧预览路径越界而失败，没有重写路径或放松containment；中断的collector也未补造终态。最终通过仅认证同一保存目录的fresh恢复，不认证任意移址、其它模型/素材/组合，或真人身份/口型质量。新通用配对工作流交付与最后人审仍独立记录。
