# H24：已审来源工作流

三份可拖入ComfyUI的前端JSON，另有[独立API图](../../../workflow_api/h24-reviewed)。[索引](index.json)记录JSON及接受片源SHA；不把作者机器路径写进公开图。

| 用途 | 工作流 |
| --- | --- |
| 参考图＋Compact独立专家4步 | [Compact_PTD_4步](H24_Compact_PTD_4步.json) |
| 生成可连续播放的音画循环 | [周期音画循环_4步](H24_周期音画循环_4步.json) |
| 跟随结构视频动作并合入原声 | [Solar_结构重渲染_4步_原声](H24_Solar_结构重渲染_4步_原声.json) |

## 资源设置

所有底模、CLIP、VAE及LoRA在对应加载节点自行选择；个人图片、视频和音频不随包分发。图里的`H24_...`素材名是待替换槽位。

- **Compact**：固定来源为[nynxz权重](https://huggingface.co/nynxz/MiniMax-H3-PTD-LoRAs/tree/d3a95b93196480b1570473ea2d1cfdd53d4fef72)中的`ptd/ptd_h3_ref2va_4step_int8.safetensors`，放到`ComfyUI/models/ptd/`。它由PTD专用节点读取，随后使用其sampler和sigmas。图中选用Ref2VA INT8底模和一张参考图。
- **循环**：图中选用FL2VA INT8与EMA B加速LoRA，4步、双时钟12/3。循环设置在该示例明确开启；新建节点默认关闭。单圈102帧/4.25秒，审片中的三圈用于检查接点。
- **Solar**：[固定权重](https://huggingface.co/Enderfga/SolarRenderer/tree/c5512be825f297ed8db390cbeda0289fb0c24779)的`solarrenderer_4step_lora.safetensors`与`solarrenderer_control_lora.safetensors`放到`models/loras/`，按图中顺序各加载一次。图中底模为Ref2VA INT8，24fps结构视频与同源首帧、LCM4步。
- **Solar静音文件**：从[Solar固定源码](https://github.com/Enderfga/SolarRenderer/tree/00b89093234a194bb06e12b7df0d2ed354769a8d)取得`h3_silence_153_158_170.safetensors`，放到`models/solar/`，或在节点中明确选择本机绝对文件。原声只用于最后合成，保留显式−1dB音量节点。RandomNoise与静音作用域共享一个种子。

底模、CLIP和两VAE的实际文件名保留已审图中的选择，换用自己的文件时注意真实模型及张量要求。资源的SHA及许可见[来源说明](../../../THIRD_PARTY_NOTICES/H24-SOURCES.md)。权重不会随打开工作流自动下载。

2026-10-27，Compact、循环和Solar三组的画面/声音均获用户完整看听接受，循环接点评为自然。公开图保留数值与接线，只适配两处本机权重路径及来源标记；未为整理工作流重复采样。来源图的浏览器保存、服务重启及原生API导出已核对。其他素材、组合和性能不继承这三份固定样片的资格。
