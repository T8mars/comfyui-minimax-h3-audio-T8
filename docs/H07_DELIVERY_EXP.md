# 作品溯源侧车（EXP）

可选独立输出节点绑定真实MP4和原生工作流的SHA256；不修改旧导出器，不重编码视频、替换声音或自动上传。默认只报告，显式确认后才在output/T8/TakeSidecars新建私有与分享JSON，各自create-only保存。

输入project／shot／take／recipe使用简短ASCII ID。音源显式声明角色ID、实际文件与SHA，origin区分original、native_generated、postprocessed；参考合成声不能冒充人声原录音。声明不等于声纹匹配或口型批准。

可连接实际完成Quality Stage或modular Stage（二选一）；沿用原producer／tensor／receipt验证，报告实际已记录NFE。Stage完整性不证明这份视频由它解码，更不自动授权缓存复用。无Stage时NFE为unknown，不从SIGMAS长度猜。

私有副本保留本机路径。分享副本只允许ID、媒体／工作流／音源哈希与大小、限定生产者字段；不复制原工作流或receipt内容、提示词、文件名、本机缓存路径、凭据或GPU ID。用户自选ID仍应在分享前检查，不因名叫分享副本就承诺匿名。只移除sidecar敏感字段，不改MP4原内嵌metadata；若要公开原片，需独立确认原片隐私。

路径／SHA变化、矛盾生产者或不支持的音源声明直接报错；不绑定另一个take来掩盖。临时tensor／trimmed video不是原文件，不偷偷重编码为可绑定文件。验收以已有真实5秒成片的原字节为准，0新采样；合同测试中的tiny fixture仅证明绑定和隐私，不是媒体或质量资格。

指定三节点CPU图已另存、关闭重开、从画布Run完成：实际5秒MP4、既有原生生成的参考声与原生工作流SHA保持，读取真实HIGH3 Stage产生两个新JSON，分享副本只有限定字段。默认不写的合同与错误SHA／矛盾producer合同分别验证；不把这次写JSON叫成片质量或缓存批准。
