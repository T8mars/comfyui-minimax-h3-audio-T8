# FL2VA × Ref2VA混合模型

这一组使用小型、可审计的AdaLN切片artifact，在FL2VA基座上研究Ref2VA参考能力，避免运行时同时装入两套完整模型。

## 推荐顺序

1. `Hybrid_Compatibility_Audit`：先核对模型族、key、shape、dtype和curve。
2. `Hybrid_Artifact_Maintenance`：构建、检查或维护小型artifact。
3. `Hybrid_Model_Advanced`：加载基础混合recipe。
4. Audio/Mixed Reference用于特定参考模态；VBAR Headroom仅用于显存策略实验。

## 当前成果

pruned FL2VA/Ref2VA对的curve-aware重基、小补丁ModelPatcher和严格指纹合同已实现。先前盲测中多数差异很小，真人素材B略有偏好但不足以证明“质量与参考能力两者兼得”。

## 使用方法与注意事项

只使用报告确认兼容的同族模型对；full与pruned禁止混用。顺序固定为原生FL基座 → Hybrid patch → LoRA → 其他wrapper。未知模型、LoRA冲突、指纹不符必须停止，不能按文件名猜。VBAR只管理权重页，不保证激活、VAE或attention永不OOM。
