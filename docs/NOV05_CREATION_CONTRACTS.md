# 素材引用与异步回写组合验证

本轮沿用现有导演台、场景 revision 和任务归属机制，没有重写队列或增加确认步骤。
以下通过的是软件消费者检查；训练后 Qwen 的识别质量与最终视频画质另行验收。

- 同一素材 ID 更换实际内容、重排引用或切镜后，迟到编译不能覆盖新预览。
- 旧任务完成后，take 仍归属原镜头；当前镜头的逐字对白和引用保持，无重复提交。
- 实际注册的 PNG/WAV 经原生 Core LoadImage、LoadAudio、TrimAudioDuration，
  再经现有项目编译与 UI 序列化，Picture 编号对应真实素材次序。
- 双空格、换行和中文原文对白保持；UI 可以重新编号节点，验收比较实际输入连接、
  素材 widget 和输出槽，不要求 UI 节点 ID 与 API 字典键相同。
- 六种现有 HyperFlow 配置经真实 Core validate_prompt；验证子进程禁止 CUDA 初始化，
  仅屏蔽与本次 CPU 计算无关的可选 GPU capability 探测。

`tests/director_nov05_regression.mjs` 使用真实会话模块与明确的 HTTP/storage 替身。
新组合和既有 JavaScript 回归合计 171 项通过。
`tests/test_nov05_director_contract.py` 与有关导演回归共 112 项通过，主进程及六个
Core 验证子进程均未初始化 CUDA。名字解析使用已有测试替身，不声称加载了训练权重。

`tests/test_director_nov05_browser.py` 提供隔离浏览器的双标签、迟到编译、切镜、逐字
对白、项目导出和刷新组合；统一集成已实跑隔离 Chromium，1 项通过，CUDA 未初始化。
浏览器 HTTP 使用明确的受控响应，验证真实 UI 事件、下载及标签状态，不冒称远端服务或
训练后生成。浏览器结果独立于上述模块测试。
本次发现的路径深度假设与子进程 CPU 防护问题只修改测试脚手架，没有发现需要猜修的
导演产品代码错误。未知用户模型和有序 LoRA 的现有执行策略保持。
