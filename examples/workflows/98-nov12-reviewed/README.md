# 已审创作工作流（v1.106.0）

22份实际接线各有UI和API副本，保留对应通过样片的模型、seed、步数与声音设置。指定样片通过不保证任意素材都能复现效果。

先选择自己的合法输入和已下载模型，再运行；本目录不附人物图、视频或模型权重。文件名里的EXP表示独立实验入口，不改变原有工作流默认值。

- **翻书4+4声音修正版**：`2D_Tier1_4plus4_Audio_Clock_EXP`。LOW采样器接声音续采节点的`coarse_sampler`，修正直接双时钟部分音频的缩放。
- **Flash双采推荐本次6+3**：完整LOW6先完成声音，再HIGH3保留这份声音，实际9次DiT。原3+3声音失败版本不在此目录，也没有被改名冒充通过。
- 五视图示例使用五张独立参考及共享单人首帧，不用拼版猜裁；请分别替换各图片输入。
- `Voice_Carry_A_to_B`包含A8+B8、真实首帧与音频尾段接续，不把两个窗误称总共8步。
- IC例只验证所示0.6MP/up3/3NFE，保留草稿原声；不是2K或默认更好的保证。Stage复用例首次8NFE，命中时0NFE，内容校验和解码仍有成本。

Flash需另装[官方独立消费者](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash)，不把Flash当普通加速LoRA。2D权重放`models/latent_upscale_models`，其余模型及可选资源见[接线说明](../../../docs/NOV12_CREATION_EXP.md)。LoRA区间样例的所测INT8组合主内存开销较高，不代表16GB显存机器的主内存也足够。

|工作流|实际模型前向|UI|API|
|---|---:|---|---|
|2D Tier1 4plus4 06MP EXP|8|[打开](2D_Tier1_4plus4_06MP_EXP.json)|[API](api/2D_Tier1_4plus4_06MP_EXP.api.json)|
|2D Tier3 4plus4 06MP EXP|8|[打开](2D_Tier3_4plus4_06MP_EXP.json)|[API](api/2D_Tier3_4plus4_06MP_EXP.api.json)|
|3D 4plus4 06MP Control|8|[打开](3D_4plus4_06MP_Control.json)|[API](api/3D_4plus4_06MP_Control.api.json)|
|DMAD Renoise8 06MP EXP|8|[打开](DMAD_Renoise8_06MP_EXP.json)|[API](api/DMAD_Renoise8_06MP_EXP.api.json)|
|PromptEdit Original 06MP EXP|8|[打开](PromptEdit_Original_06MP_EXP.json)|[API](api/PromptEdit_Original_06MP_EXP.api.json)|
|PromptEdit Rewrite 06MP EXP|8|[打开](PromptEdit_Rewrite_06MP_EXP.json)|[API](api/PromptEdit_Rewrite_06MP_EXP.api.json)|
|PromptEdit Guidance 06MP EXP|14|[打开](PromptEdit_Guidance_06MP_EXP.json)|[API](api/PromptEdit_Guidance_06MP_EXP.api.json)|
|LoRA Windows 06MP EXP|8|[打开](LoRA_Windows_06MP_EXP.json)|[API](api/LoRA_Windows_06MP_EXP.api.json)|
|IC Draft UP3 06MP EXP|3|[打开](IC_Draft_UP3_06MP_EXP.json)|[API](api/IC_Draft_UP3_06MP_EXP.api.json)|
|IC Off Control 06MP EXP|3|[打开](IC_Off_Control_06MP_EXP.json)|[API](api/IC_Off_Control_06MP_EXP.api.json)|
|Flash FL6 06MP EXP|12|[打开](Flash_FL6_06MP_EXP.json)|[API](api/Flash_FL6_06MP_EXP.api.json)|
|Dual 3D Book Motion EXP|8|[打开](Dual_3D_Book_Motion_EXP.json)|[API](api/Dual_3D_Book_Motion_EXP.api.json)|
|Dual 2D T6 Book Motion EXP|8|[打开](Dual_2D_T6_Book_Motion_EXP.json)|[API](api/Dual_2D_T6_Book_Motion_EXP.api.json)|
|Voice Carry A to B EXP|16|[打开](Voice_Carry_A_to_B_EXP.json)|[API](api/Voice_Carry_A_to_B_EXP.api.json)|
|Flash Ref6 06MP EXP|12|[打开](Flash_Ref6_06MP_EXP.json)|[API](api/Flash_Ref6_06MP_EXP.api.json)|
|Paired PCM Windows New Source EXP|6|[打开](Paired_PCM_Windows_New_Source_EXP.json)|[API](api/Paired_PCM_Windows_New_Source_EXP.api.json)|
|Native Stage Reuse 8 EXP|首次8 / 命中0|[打开](Native_Stage_Reuse_8_EXP.json)|[API](api/Native_Stage_Reuse_8_EXP.api.json)|
|Five References Turn Hybrid8 06MP EXP|8|[打开](Five_References_Turn_Hybrid8_06MP_EXP.json)|[API](api/Five_References_Turn_Hybrid8_06MP_EXP.api.json)|
|Front Reference Turn Hybrid8 06MP Control|8|[打开](Front_Reference_Turn_Hybrid8_06MP_Control.json)|[API](api/Front_Reference_Turn_Hybrid8_06MP_Control.api.json)|
|2D Tier1 4plus4 Audio Clock EXP|8|[打开](2D_Tier1_4plus4_Audio_Clock_EXP.json)|[API](api/2D_Tier1_4plus4_Audio_Clock_EXP.api.json)|
|Flash FL 6plus3 Complete Audio EXP|9|[打开](Flash_FL_6plus3_Complete_Audio_EXP.json)|[API](api/Flash_FL_6plus3_Complete_Audio_EXP.api.json)|
|Flash Ref 6plus3 Complete Audio EXP|9|[打开](Flash_Ref_6plus3_Complete_Audio_EXP.json)|[API](api/Flash_Ref_6plus3_Complete_Audio_EXP.api.json)|

[验收范围与实际媒体SHA](../../../docs/NOV12_ACCEPTANCE.md)。ASR、原生27B文字LoRA和RGB对照分别保留自己的机制/实验边界，不借这些视频的评价获得通用质量资格；私人写作指南和失败工作流不公开。
