# Windows 路径与示例命名

## 仓库约束

- 所有仓库相对文件路径最多 **140 个 UTF-16 字符单位**，包括目录、分隔符和扩展名。
- `examples/workflows` 内的 JSON 文件名最多 **96 个字符单位**；建议新文件名控制在 64 内。
- 推荐 `Sxx_功能_配方_Full或Cold_EXP.json`。日期、每次迭代和全部效果说明放 README／画布 Note，不叠加到文件名。
- 不允许 Windows 保留名、非法字符、尾随点／空格或仅大小写不同的文件路径。
- `.github/workflows/windows-paths.yml` 在 Linux／Windows 上检查实际 Git 文件列表；超限阻止该检查通过。开发时用 `python tools/check_windows_paths.py --working-tree`，真实 ZIP 用 `--zip 包路径`。构建器输出必须遵守同一预算，不能生成完又加长后缀。

这是有限路径预算，不保证任意深的安装目录都能工作。插件仓库根绝对路径不超过 110 字符时，140 的相对预算还预留 8 字符临时后缀，并在经典 MAX_PATH 的 260（含终止符）以内。目录更深时应缩短安装路径。参见 [Microsoft 的路径限制说明](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)。本补丁不修改注册表或全局 Git 配置。

## 旧示例在哪里

37 个过长示例只改文件名，JSON 原字节、节点、接线、采样参数、声音与缓存字段不变。完整 [旧名→新名与 SHA256 对照](../examples/workflows/filename-map.tsv) 随示例保存，不能为了旧链接再放回超长文件或创建别名副本。`tools/workflow_paths.py` 仅帮助已有构建器定位新的公开文件名，不改 ComfyUI 执行行为。

用户自己已经保存的画布、`user/default/workflows`、缓存、成片和模型文件不重命名。历史备注中的旧名称是来源记录，不是失效的模型或必须重跑的信号；公开 README 链接使用新名称。

对照表SHA256按仓库 `.gitattributes` 声明的LF表示记录；本地旧CRLF和线上LF仅按这项换行规则比较，不做JSON重排／删字段／改参数。本地重命名仍保持各文件实际原字节。

## 已经卡在 Filename too long 的更新

先备份本地修改。若旧路径让现有克隆无法完成切换，可在**该插件目录**执行一次：

```powershell
git -c core.longpaths=true pull --ff-only
```

此开关只用于这次命令，不写全局设置。仓库有本地修改、冲突或非快进时先处理并保留修改，不使用 `reset --hard`／删插件／强制覆盖。Manager 的 Git 与命令行可能不是同一个程序；必要时用其实际 Git 执行。该命令只有短路径补丁已进入 main 才能取得修复；发布前不能宣称普通更新已恢复。更深的安装目录仍需迁移到短目录。
