# H05 C01：注意力存储地址只读检查

`tools/inspect_h3_attention_storage.py` 是诊断工具，不是新注意力节点、
Sage 内核适配或准入禁令。旧 KJ／Sol 委托、contiguous 策略、采样数学及
未知用户组合保持；工具不安装、执行或替换内核，不复制张量或读取数值。

Python `describe_tensor(q, layout="NHD")` 读取实际 PyTorch tensor 的 shape、
stride、storage offset、底层 storage 字节数、dtype 和 element size。
NHD 指 `[batch, sequence, heads, dimension]`，HND 指
`[batch, heads, sequence, dimension]`。不知道布局时显式使用 `unknown`，
不能从相同形状推断布局。输出使用相对存储位置，不泄露进程指针地址。

“实际张量”还要求实际 backing storage 不是 meta。PyTorch FakeTensor 可声明
CPU／CUDA 设备，却仍使用虚拟 meta storage；这类编译／形状模拟张量不当作真实
QKV 存储证据。检查实际 storage.device，不按类名封禁所有张量子类；真实有
backing storage 的 Parameter 仍能使用。该拒绝仅属于诊断工具的证据边界，
不会阻止用户普通采样、torch.compile 或未知 backend 委托。

报告分别给出最后 sequence 行的相对 element offset、实际 view 的最后
storage element 和排他末尾字节，以及各自是否到达 signed-i32 边界。
字节与 element 不能混为一个判断；非零 offset、非连续 view、广播 stride
和空 view 分别处理。报错只指输入描述符／真实存储越界，不替用户切换算法。
空 view 不访问任何元素，首／末元素和排他末尾字节均为 null；PyTorch
允许其声明非零 storage offset，不能因此误报一次并不存在的内存访问。

CLI 接受不超过64KiB的元数据 JSON，只读一次，不读 tensor／模型／包内
执行配置。例如 `python tools/inspect_h3_attention_storage.py descriptor.json`。
顶层字段为 `layout`、`backend`、`source_sha256`、`tensors`；后者仅 q/k/v，
每项为 `shape`、`stride`、`storage_offset`、`storage_bytes`、`dtype`、
`element_size`。JSON和其中的 backend／SHA 都是声明，不是 live tensor、
真实 callable 或实际加载内核来源证明。

该检查不认证 padded sequence、head group、量化临时缓冲区或具体内核内部
地址计算，也不以 view 小于2³¹证明内核安全。CUDA 内核 pin、实际入口张量、
device/SM、padding/quantizer 合同仍需在确有问题的所选路径上单独取证。
不做208K／9GB GPU探针，不修改第三方源码，不替换未知 backend。

本地资格只涵盖小 CPU tensor 的实际元数据、不分配大 tensor 的整数边界
检查及有界 JSON reader；不代表本机 Sage GPU／SM120／B200 已完成验证。

另新增 3 个 CPU 回归：FakeTensor 工厂、真实张量的 fake 视图不被认证，真实
Parameter 不被误禁。旧工具在前两个反例上确实失败，修正实际 storage 检查后
同一 3 项通过；没有重跑历史 18 项，也不把不同源码 epoch 的计数相加。
原 describe_storage 地址数学、常量、JSON reader 与 CLI 的 AST 保持不变，
无 CUDA 初始化／GPU 探针／内核执行／新注册节点或采样改动。
