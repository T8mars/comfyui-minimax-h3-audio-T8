# v1.93.0：RADAR r6 可组合参考、配方卡与交付工具

新增七个可组合EXP节点：六个参考包入口、一个已保存master的后处理保存出口。当前663个唯一ID，旧656个完整schema、顺序、默认及已有工作流保持。没有更换采样步表、音频时钟、3D放大器或Temporal Chunk对白方案。

## 新功能

- **参考包Create／Save／Load／Route／Apply／外置Relay**：从自己的合法图像、视频或音频创建内容与实际producer绑定的安全包，存入配置的`models/refmods`；显式有序角色可分别撤视觉、保留画外声音参考。重新构建Qwen条件，VAE latent复用，不直接编辑旧embedding。EAV／Prompt Relay仍可外置。
- **模型配方卡与差异确认**：普通模型／VAE／LoRA／Bridge节点右键查看所选已安装资源的有界header证据；训练alpha／rank、运行strength及Bridge alpha分开显示。unknown不猜、不禁原加载器；明确预览和确认才修改支持的字段，可撤销，不自动保存或排队。
- **精确最终画幅与原片救援**：在全部回贴完成后先保存master，再用Core裁切／留边及独立后处理出口。声音从master逐包复制；失败阻断新成片出口，原片保留，不覆盖旧文件。限定完整零起点24fps SDR，不偷换VFR／HDR／时间线。
- **导演台连续性小配方**：现有“镜头证据与规则”增加角色道具连续账本、画内／画外对话。只填空表单，仍走已有快照、待审候选、明确采用／回退／保存，不自动改事件、声音接线或推断事实。
- **LightVAE解码示例**：普通VAELoader从`models/vae`加载固定Kijai Light转换件，接既有T8 AVDecode的视频插口，音频VAE不变；无需新loader、Core升级或Hyper2×倍率。

## 工作流

[75-radar-r6-reference](../examples/workflows/75-radar-r6-reference/README.md)有10张原生保存来源的短名公开图：参考编码、Full／HIGH-only Cold采样、FullVAE／LightVAE独立解码、三种master模块和两张配方卡UI图。私有素材、参考包、Stage及实际缓存SHA不随仓库／包提供，选择项必须填写自己的真实输入；不使用占位缓存排队。

[74-radar-r6-delivery](../examples/workflows/74-radar-r6-delivery/README.md)另有三张通用master后处理模块。Full是生成之后的附加模块，不是另一套生成引擎；Cold只处理已保存master。

## 验收与限制

五个指定代表均已获用户人工通过：参考包原生4+4生成后的5秒成片、LightVAE对同一完成HIGH的解码对照、配方卡UI、导演台两条规则UI、原片／裁切对照。Light与普通解码的完整decoded PCM和视频PTS一致，未重采或替换音轨。

参考测试两包来自同一份素材，不证明两个人身份／声纹锁定。演示采样producer绑定`--cpu-vae --fp32-vae`；匹配配置或从自己的合法素材重新建包，不忽略身份校验。该Core的INT8 CPU解码尾部失败，正确采样图只保存完成Stage，另用正常GPU VAE图成功解码；不把原失败整图当success。Cold未追加HIGH GPU，Full／Contain等模块仅模板资格；EAV是`report_only`，不宣称增强质量。

LightVAE固定转换文件2,137,493,384B／SHA256 `f11f8b9b96c9dd22b36bc6c18464b6929c207d247e6d865d90fe31588e58e7e2`，实际Core26层decoder／24通道严格加载无缺失或多余权重。请自行阅读并遵守原作者许可；公开可下载不免除许可。本次只认证指定decode对照，不认证encoder、全部素材、声纹、Hyper2×或通用速度。

QuantFunc／Nunchaku不恢复；条件量化栈、社区格式互操作等没有未经资格的实现混入。模型、素材、缓存、私有交接与审核文件不上传。GitHub正式发布不等于Comfy Registry已经Active／可安装，Registry状态单独处理，不阻塞本次GitHub交付。

详细合同：[总览](RADAR_R6.md)、[参考包](RADAR_R6_REFERENCES.md)、[后处理](RADAR_R6_DELIVERY.md)、[导演台规则](RADAR_R6_CONTINUITY.md)。
