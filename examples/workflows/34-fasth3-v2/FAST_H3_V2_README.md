# FastH3 V2：六份通过配方

完整学生模型、安装与组件见[FastH3 V2](../../../docs/FAST_H3_V2_EXP.md)。旧255图未改，旧失败Sol A不进入推荐。全部保持EXP，仅绑定样片及已评分项目接受。

- [训练VSA h1/c1](FastH3_V2_Trained_VSA_73f_h1c1_EXP.json)：832×480、73帧、训练DMD8、min_tokens=0，本机已审基础控制。
- [训练VSA h4/c2](FastH3_V2_Trained_VSA_73f_h4c2_EXP.json)：同长度与配方；本机更慢且周期整卡占用更高，不作通用提速／省显存推荐。
- [官方Comfy模板](FastH3_V2_Official_Comfy_Template_124f_EXP.json)：124帧、simple8/res_multistep、keep10%、起始20%；不是训练DMD配方。
- [Dense Relay双MODEL4+放大+4](FastH3_V2_Dense_Relay_Dual_4plus4_8s_EXP.json)：首帧2:3，256×384→原有learned3D→512×768，总8秒，保留global/local/时间线。
- [训练VSA双MODEL中断恢复](FastH3_V2_Trained_VSA_Dual_4plus4_Resume_8s_EXP.json)：同2:3尺寸和4+4；已审真实中断后由新进程生成第二段，Relay/EAV关闭。
- [Sol音频／文本精确保留](FastH3_V2_Dense_Sol_Audio_Protected_73f_EXP.json)：已认证外部Sol选择器＋Dense专用保护，73帧、tau0.5；B已明确接受，不加音量、不换音轨。

单MODEL图保持已审min_tokens=0，节点全局缺省12288不改；诊断观察器移除后转录同一Core RandomNoise→BasicGuider(CFG1)→SamplerCustomAdvanced连接，解码最终sigma0输出。不是重新GPU生成或位级复现保证。

双MODEL保留两个独立加载器，可分别接普通权重LoRA；不要套旧EMA/Turbo加速LoRA，任意完整模型LoRA质量未认证。window124/context22时接缝约5.17秒，两段总8秒不是每段4秒。执行前换新chain_id，只有参数不变才resume，改模型／VAE／LoRA／参考／代码后不搬旧缓存。

首帧引用已审本地图，请放入ComfyUI/input或用Load Image选自己的2:3图，不拉伸；更换参考是新配置，不保证相同质量。没有隐式模型下载。Sol需已安装匹配的外部Sol组件与CUDA内核；保护只针对Dense无Relay，Relay另保留时间bias，旧PyTorch/KJ／训练VSA／模板路线不改。

六图真实原生UI与API往返已核验。固定832×480／73帧的生产配方冷／热对照已完成：旧EMA-B原生8步整图103.35／99.90秒，V2训练DMD8为78.20／71.92秒；均实际8次执行／400次选定后端调用，完整H.264+AAC解码通过。仅这组V2更快，整卡显存观察未下降；不是同模型或等画质证明，不承诺普遍更快、更省或通用16GB安全。完整测量范围见上方FastH3 V2说明。

六份控制图随 v1.83.0 提供；见[版本说明](../../../docs/RELEASE_1.83.0.md)。轻微接缝变色为已知限制。人为帧数上限已移除，但超长 tensor 可能耗尽显存；本次资格不延伸到24／32秒，不影响旧工作流。Registry 状态以实际发布验证为准。
