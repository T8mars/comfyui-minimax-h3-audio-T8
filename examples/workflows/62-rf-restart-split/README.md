# S29 RF Restart 分离节点（实验图）

三条入口 `standalone`、`detail_mixer`、`two_pass_detail_mixer`，每条各有 `minimal`、`effects`、`save_effects`、`resume_effects` 四图，共 12 张。BASE 下降、显式 Handoff、RESTART 下降是可独立查看与连接的节点；`resume_effects` 从 Stage Load 读取之前保存的 BASE，不重新运行 BASE。效果图外置 EAV／Prompt Relay 及相应时钟控制。旧 RF、Mixer 和一体工作流未改。

恢复图的 `artifact_path`／`artifact_sha256` 是必须替换的占位值；还需匹配冻结片段长度、条件布局和原 sigma 模板。默认值不能直接作为成功恢复。当前有保存 JSON、确定性 CPU 阶段合同及12张公开图的原生浏览器 Save As／重开资格；三阶段保存图与只恢复 RESTART 图还分别通过可见 Relay 与末段 EAV 控件编辑、另存重开。尚无这些公开图的真实权重视频与音频、旧新数值同一性或人工验收。其它双采路线的效果外置也不能从本目录推定已完成。
