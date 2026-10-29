# Open-H3 Regenerate 2K 接入（实验）

## 完整作者配方

新增 **H3 Open Regenerate · 完整配方加载**：读取作者的合并线性LoRA及独立Sharp曲线分支。普通LoRA加载器只能加载线性权重，不能代替独立时间曲线的计算。

- [原项目](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K)、[项目演示与说明](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/)、[作者权重](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)。
- 下载 `lora/open_h3_regenerate_2k_lora.safetensors`（约12.45GB）及 `lora/sharp_curve_branches.safetensors`（约1.24GB），放到 `models/loras/OpenRegenerateAuthor/`。
- `curve_source_name` 选择含 `adaln_t_table` 的作者Ref2VA pruned来源。本次使用本地 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`，只读其中1025×8表，不再加载一套Ref2VA底模到GPU。
- 合并文件包含Refiner384、LightX2V Turbo v1.2与LMS Sharp线性权重，Turbo的alpha/rank已经折入B。两个强度默认均1，不重复除rank或再次叠一遍同款加速LoRA。
- 独立Sharp分支在50个AdaLN投影上计算 `base(x) + strength × B(A(curve(t)))`。曲线使用实际原生时间，保留上游用户MODEL补丁、LoRA顺序及强度。

**作者默认：Euler、CFG1、3步，sigma为 `[0.909375, 0.8571428571428571, 0.6666666666666666, 0]`。** 这是实际视频sigma，准备节点设时钟1/1，不能再次套12/3。3步是完整配方的发布默认，不能据此断言单独Kijai纯refiner的3步必然有同样效果。

两种入口已保存为原生前端工作流：

1. [现成视频 → 完整配方2K细化](../examples/workflows/92-open-regenerate/CompleteRecipe_ExistingVideo_2K_EXP.json)：默认1024×576输入，学习型3D潜空间放大到2560×1440，再用完整配方3步。原音从来源窗口单独送保存；无音轨时明确生成静音。
2. [HyperFlow8 → 完整配方3步2K](../examples/workflows/92-open-regenerate/CompleteRecipe_HyperFlow8_Refine3_2K_EXP.json)：T2VA一采1344×768，完成潜空间直接接学习型放大和二采，无中间视频重编码。最终保留一采生成的原声音轨。只需一个共用底模加载节点，分别派生HyperFlow专用MODEL分支及完整配方MODEL分支；不把HyperFlow端点补丁带到普通细化分支。

两图默认原生124帧网格、交付121帧/24fps，约5.04秒。准备节点在模型里编码双声道静音并锁定音频，最终声音独立送保存，不拿模型里的静音替换原音。现成视频不足原生网格的末尾明确保持末帧。

**H3 Open Regenerate · 实际运行报告** 的 `after_sampling` 接实际采样后的LATENT，报告观察到的模型前向、50个分支调用及时间对齐情况；原样返回LATENT。加载完成仍是“未观察采样”。计数为该MODEL准备以来累计，不代表另一个缓存LATENT、CUDA graph重放或人工画质通过。

更多步数可选择 `custom_shift6`，再设置 `custom_steps` 和 `start_sigma`；另外两张发布表使用各自固定sigma，不受这两个自定义参数影响。作者还给出较柔和的2步表 `[0.7, 0.49411764705882355, 0]`。多步或低起点均不保证消除源片已有的运动模糊。

现成视频的默认完整3步路线已在完整训练模型上生成2560×1440、121帧/24fps；实际观察3次前向、50个Sharp分支各调用3次、208个线性目标和原生时间对齐。所选原片没有音轨，明确使用静音，不冒称验证了该片的原声保持。首次执行约3700.20秒，Torch峰值已分配约11.71GiB，非整机总显存。

两种入口及更多步数对照均已完成所选完整训练模型实测：默认3步已有视频细化、HyperFlow8+3直接潜空间双采、同源8步/起点0.7实验。三组均交付121帧2K；双采另保存1344×768一采片，二采保存前的声音张量与一采逐样本相同。无音轨的已有视频两组明确使用静音。

两个完整图及8步实验图均在独立原生前端完成加载、保存、重开、导出，显式参数和连线完整保留。三组对照已完成实际浏览器整片播放、同步定位、逐帧、局部对照、半速、全屏及声音单选，自动验证时反馈缺省保持“未审”；2026-10-29用户直接确认当前三组样片“没问题”并授权推送GitHub，人工接受单独绑定媒体和工作流。83项新功能/内存回归与144项独立兼容回归通过，包含本地原始HyperFlow权重检查；另复核2项节点注册/元数据测试，不重复计入227项。

本机使用原生Core/INT8底模、NVFP4文本编码器、8组注意力及8块FFN。后两组每次实际前向均核对50层的8块FFN；普通细化每前向还观察到50次分组注意力调用，HyperFlow专用前向保持自身实现。三组的208个线性目标、50个Sharp分支调用和原生时间对齐均核对通过。首次默认3步来自所选bare模型的初版实现；后两组使用修正父层委托及共享分支内存配置的当前实现，不能作跨版本公平速度对比。

| 所选路线 | 实际前向 | 本机整图执行秒数 |
| --- | ---: | ---: |
| 已有视频完整3步 | 3 | 3700.20 |
| HyperFlow8＋完整3步 | 11 | 3500.59 |
| 已有视频8步／起点0.7实验 | 8 | 6290.06 |

这些是所选配置的测量，不是通用容量或速度认证。8步实验同时改变步数和起点，不把差异仅归因于步数，也不作为作者默认推荐。当前三组指定样片已获人工接受；不扩大为源片运动模糊、手部或身份变化的通用修复保证。

本实现使用原生Core采样与现有H3 VAE，不包含作者的SoL-Engine或StreamRender-H3快速VAE部署。作者也提供原版VAE的标准路径；本地“完整配方”指完整权重、独立曲线及细化流程，不意味着具有其GB200加速后端。

共享底模的分支在配置内存优化时读取所选MODEL的逻辑原方法及Core备份，避免把另一活跃分支的分块方法再包装一次。Sharp分支只在实际模型前向期间安装临时hook，成功或异常退出均移除，保留用户选择的父层替换及上游hook。新增回归核对真实分块次数、数值及切回裸模型的恢复；不能把配置标签当成实际运行证明。

本次作者合并权重SHA为 `8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04`；独立曲线文件为 `26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e`；曲线表张量为 `c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7`。这些用于绑定验证证据，不是限制用户换模的白名单。

## Kijai纯Refiner入口

Kijai 的 `minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors` 是 **refiner LoRA**。放在 `models/loras/`，使用现有 **MiniMax H3 LoRA Compatibility Loader** 或原生 **Load LoRA Model Only**，强度从1开始；无需新增专用加载器或再次转换。

不安装新增准备节点也能接线：[仅使用既有节点的工作流](../examples/workflows/92-open-regenerate/RefinerOnly_ExistingNodes_2K.json)。它使用原生Load LoRA Model Only、既有Source AV、双时钟采样设置和ManualSigmas，已核对已发布节点的浏览器保存/重开/导出。新增的细化准备节点只是简化接线的可选入口，加载这个LoRA本身不依赖它。

**既有节点路线已做完整训练模型实测。** 使用已发布的`00e875d`源码快照、FL2VA pruned INT8底模、原生LoRA加载器及所选纯refiner强度1，分别完成576×1024和1440×2560两段4秒/96帧/3步视频。均完整解码，原声音频在有损封装前逐样本一致。无需为这个文件更新节点；需要按下文细化接线使用，不能只把它替换进任意旧4+4配方就宣称等效。

文件为原生 Comfy H3 前缀，416个BF16张量、208组线性适配。Q/K/V已融合，融合投影rank为1152，其余为384；alpha及强度已折入B，不再手动除以384。当前所选文件完整校验值为 `4bf30291343eef5e51ee867ba37dd31bce3133d1da1ace1a245a7130b88818c2`，大小5,868,409,512字节。校验用于本次验证记录，不是节点的模型准入条件。

## 与作者完整配方的区别

[Kijai 文件](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors)的元数据明确说明它只截取原合并权重的refiner部分。它不包含Turbo或Sharp。

[NVIDIA Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)的完整发布配方合并了refiner384、LightX2V FL2VA Turbo v1.2和LMS Sharp，还单独加载Sharp曲线分支。单独加载Kijai这个文件，不等于完整配方；仅叠加普通Sharp线性权重也不能证明曲线分支已生效。节点不会自动下载、叠加或替换这些权重。

## 接线

1. 使用独立的细化MODEL分支，加载自己选择的底模及refiner。原配方使用FL2VA；其它模型保留用户选择，但不冒称已验证。不要直接复用带HyperFlow专用端点补丁的一采MODEL当作普通细化MODEL。
2. 使用完整完成的一采潜空间，或把现成视频用H3视频VAE编码。复用已有学习型3D潜空间放大器，目标可设2560×1440。使用当前源潜空间坐标；不把普通插值说成学习型放大。
3. 连接 **H3 Open Regenerate · 细化采样准备**：MODEL、视频潜空间及Audio VAE。该节点编码真正的双声道静音作为模型上下文，视频mask为1，音频mask为0；输入对象不被修改。静音不是全零音频latent。
4. MODEL输出接 **BasicGuider**，使用原镜头的普通提示词；sampler/sigmas接 **SamplerCustomAdvanced**。学习型放大器保持时长不变，因此准备节点可以在其前或后；采样器的latent输入必须接最终放大的潜空间。
5. 解码细化画面，原视频AUDIO独立接裁切和保存节点。模型里的静音上下文不作为成片声音。

已保存[纯Refiner接线示例](../examples/workflows/92-open-regenerate/RefinerOnly_2K_EXP.json)。默认读取开头5秒，按124帧原生网格准备后交付120帧，原音独立裁切5秒；换成自己的视频和原提示词。该图依赖本开发候选新增节点，尚未部署到用户Core，也不是已验收的2K质量推荐配方。

既有节点版默认肖像480×832输入、1440×2560输出，4秒/96交付帧；使用8组注意力与8块FFN减少激活显存。横屏可显式修改低分辨率1344×768和目标2560×1440。原生图里直接设置video/audio shift=1/1，ManualSigmas输入上表原始sigma；SourceMediaWindow的静音送Audio VAE，Source AV中视频regenerate/音频lock，原音独立保存。旧版没有精确`frame_count`可选字段时，4秒×24fps本身就是96帧，不依赖新字段。

## 采样表

| 选项 | 视频sigma |
| --- | --- |
| published_3step | 0.909375 → 0.857142857 → 0.666666667 → 0 |
| published_2step | 0.7 → 0.494117647 → 0 |
| custom_shift6 | 用户指定起点/步数，按shift6生成 |

使用Euler和BasicGuider（CFG1）。这些已经是实际视频sigma，不能再套旧的12/3移位。新节点明确把采样时钟设为1/1，锁定音频行保持原生时间1，保留现有上游权重补丁。初始化由Comfy原生flow采样器执行 `(1−sigma)·latent + sigma·noise`，不要提前手工混噪后再让采样器重复加噪。若输入MODEL另设noise_scale，其用户设置保留，报告显示该值。

## 已验证与未验证

- 用户下载的完整文件SHA与发布LFS一致；对所选FL2VA INT8文件头，208组目标和尺寸全部匹配。
- 实际416张量通过现有原生Core加载器及H3兼容加载器的CPU检查，向真实完整尺寸的原生H3 **meta模型**登记208个目标，未匹配0。另有下述完整训练模型GPU实测，二者分开记录。
- 实际tiny Core运行了2步、3步Euler，验证首步混噪、精确前向次数、编码静音锁定和输入不变；原生加载器实际消费rank1152的融合QKV及用户强度，不重复缩放。
- **既有节点示例**的23个执行节点/30条连接已用已发布源码在独立无头浏览器完成保存、重开和导出；没有新增准备节点。完整训练模型实际执行2组，各3次前向，均交付96帧，完整视频/原音/播放/拖动/全屏已核对。
- 本机所选4秒案例：576×1024执行193.45秒，1440×2560执行1795.33秒。使用8组注意力、8块FFN、预留5GB显存及关闭编译/CUDA graphs的独立进程；2K的Torch峰值已分配显存约10.17GiB，非整机总显存。这是所选配置的测量，不是公平速度对比、任意时长或通用16GB资格。
- **可选新准备节点**已用于上文完整配方的真实训练模型3步路线；单独的18节点纯Refiner示例仍仅完成schema/序列化核对，未另做该图的完整训练GPU或浏览器验收。两种资格不混用。
- Agent抽帧对照可见人物面貌与细节会重绘，不能承诺身份完全不变或必然变清晰。完整配方默认3步、8+3及8步低起点三组已实跑且所选样片获人工接受；长视频、任意底模组合、作者BF16逐位一致及原作者加速引擎仍未验证。旧纯Refiner手部问题不因新流程能运行而撤销。

原有4+4、HyperFlow、长视频接缝、模型选择和默认设置不随此实验改变。模型许可与各合并组件来源见[来源记录](../THIRD_PARTY_NOTICES/OPEN-REGENERATE-SOURCES.md)及作者发布说明。


## 已审源码更新（2026-10-29）

用户确认当前三组对照“没问题，推送github”。[工作流索引](../examples/workflows/93-director-reviewed/index.json)记录所选视频与原生保存图的SHA。8步低起点图作为实验一并保存，默认仍为作者3步。旧Kijai纯Refiner手部失败不因本次接受而撤销；可选18节点纯Refiner图的独立验证边界也不扩大。本次保持版本1.102.0，更新GitHub源码，不重复发布Registry版本或自动重启用户Core。
