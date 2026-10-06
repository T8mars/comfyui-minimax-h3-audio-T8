# FreeVideo 新四档：Full／Cold 模板

新版Quality v2单独追加，不覆盖[旧8+2和真正4+4](../72-freevideo-split/README.md)。先按[准备说明](../../../docs/FREEVIDEO_QUALITY_EXP.md)建立独立新配置、选择真实本机CLIP／VAEs／learned放大器。Loader空值只读取专用`user/default/T8/freevideo-runtime-v2.json`；没有配置就明确失败，不暗下主体或回落旧v1。

- [Light Full 8+3](FVQ_Light_8plus3.json)：LOW8 → 外置learned1.2× → HIGH3 → 原生音画解码保存；外置LOW／HIGH Relay及EAV报告。
- [Light Cold HIGH3](FVQ_Light_Cold_HIGH3.json)：只从自己已完成的新LOW8继续HIGH3，没有LOW采样器；填写LOW Stage Save输出的真实manifest路径和SHA。
- [Single 默认Medium12](FVQ_Single_12.json)：联合AV单采12，直接解码。Quality下拉可改High16／Max20，不接HIGH3；16／20仅CPU结构支持，未新增GPU人审。
- [Medium Cold Decode](FVQ_Medium_Cold_Decode.json)：读取自己完成的SINGLE12或新HIGH终态，0扩散步，直接解码；不能把LOW8当Light最终成片。

四图保留实际原生保存的接线、布局、数学参数、编码和音频设置，仅清空私有配置／缓存路径、隐藏未执行的空预览，并写非执行资格标记。Stage路径及SHA刻意留空，必须先保存自己的Full结果；不分发作者／测试者缓存。缓存内容、源、表、clock、task、角色、帧数和音轨lineage都要符合，不能换名硬接旧MID。

本版人审仅两条固定5秒Light8+3／Medium12样片，120帧／24fps／512×288；Cold只CPU冷读取和原生保存重开，0额外GPU。Light实际448×256→1.2×的32grid放大，不是作者默认2×或普适画质／提速保证。EAV仅report_only不证明增强，Relay为beta文本种子机制EXP。主Comfy Python和原v1配置不用升级；不要把新配置指向旧计算源。
