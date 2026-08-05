# Example

`audio_lock_api.json` 是 API prompt 格式示例，展示完整链路：音频窗口、统一条件、H3
采样、联合解码、同步裁切和 MP4 输出。

使用前把四个 loader 中的模型文件名和 `LoadAudio.audio` 改为本机实际文件。示例最后使用
VideoHelperSuite 的 `VHS_VideoCombine`；核心 T8 节点本身不依赖 VideoHelperSuite。
