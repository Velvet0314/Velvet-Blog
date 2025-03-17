---
title: 一些环境配置的 Tips
createTime: 2025/01/21 15:12:40
sticky: 5
permalink: /projects/envtips
---

## 有关 WSL 的网络代理

由于 WSL 默认使用 NAT 网络模式：`检测到 localhost 代理配置，但未镜像到 WSL。NAT 模式下的 WSL 不支持 localhost 代理`，为了在 WSL 中使用代理以完成某些操作，需要对 WSL 进行一些配置。

在 Windows `%USERFILE%`目录下，新建 `.wslconfig` 配置文件，并加入以下配置：

```ini
[wsl2]                      # 核心配置
autoProxy=true            # 是否强制 WSL2/WSLg 子系统使用 Windows 代理设置（请根据实际需要启用）
dnsTunneling=true          # WSL2/WSLg DNS 代理隧道，以便由 Windows 代理转发 DNS 请求（请根据实际需要启用）
firewall=true               # WSL2/WSLg 子系统的 Windows 防火墙集成，以便 Hyper-V 或者 Windows 筛选平台（WFP）能过滤子系统流量（请根据实际需要启用）
guiApplications=true        # 启用 WSLg GUI 图形化程序支持
ipv6=true                   # 启用 IPv6 网络支持
# localhostForwarding=true    # 启用 localhost 网络转发支持（新版已不支持在 mirrored 模式下使用，会自动忽略，所以无需注释掉，只是启用会有条烦人的警告而已）
# memory=4GB                  # 限制 WSL2/WSLg 子系统的最大内存占用
# processors=8                # 设置 WSL2/WSLg 子系统的逻辑 CPU 核心数为 8（最大肯定没法超过硬件的物理逻辑核心数）
# pageReporting=true          # 启用 WSL2/WSLg 子系统页面文件通报，以便 Windows 回收已分配但未使用的内存
nestedVirtualization=true   # 启用 WSL2/WSLg 子系统嵌套虚拟化功能支持
networkingMode=mirrored     # 启用镜像网络特性支持
vmIdleTimeout=-1            # WSL2 VM 实例空闲超时关闭时间，-1 为永不关闭，根据参数说明，目前似乎仅适用于 Win11+

[experimental]                  # 实验性功能（按照过往经验，若后续转正，则是配置在上面的 [wsl2] 选节）
autoMemoryReclaim=gradual       # 启用空闲内存自动缓慢回收，其它选项：dropcache / disabled（立即/禁用）
hostAddressLoopback=true        # 启用 WSL2/WSLg 子系统和 Windows 宿主之间的本地回环互通支持
sparseVhd=true                  # 启用 WSL2/WSLg 子系统虚拟硬盘空间自动回收
bestEffortDnsParsing=true       # 和 dnsTunneling 配合使用，Windows 将从 DNS 请求中提取问题并尝试解决该问题，从而忽略未知记录（请根据实际需要启用）
# useWindowsDnsCache=false        # 和 dnsTunneling 配合使用，决定是否使用 Windows DNS 缓存池（新版已移除此实验性功能，未能转正）
# ignoredPorts=3306               # 见：https://learn.microsoft.com/zh-cn/windows/wsl/wsl-config#experimental-settings
```

完成后重启 Windows 使得改动生效

设置成功后在有代理时启动 WSL 将不再有提示信息，同时执行 `curl -I https://www/google.com` 会返回 `200` code。 
## 有关 WSL 与 SSH

通过 SSH 远程访问 WSL 实现远程办公

1. Zerotier 内网穿透

	- 在 WSL 中下载 Zerotier，并启动服务
	```bash
	sudo systemctl enable zerotier-one
	sudo systemctl start zerotier-one
	```
	- 加入 Zerotier 的网络
	```bash
	sudo zerotier-cli join <your-network-id>
	```
	- 查看 Zerotier 状态
	```bash
	sudo zerotier-cli status
	```
	- 找到当前为 WSL 分配的虚拟地址
	```bash
	sudo zerotier-cli listnetworks
	```
2. SSH 服务配置

	- 下载并启动 SSH 服务
	```bash
	sudo service ssh start
	```
	- 检查 SSH 服务状态
	```bash
	sudo service ssh status
	```
3. 修改 SSH 配置

	- 进入配置文件 `/etc/ssh/sshd_config`
	```bash
	sudo nano /etc/ssh/sshd_config
	```
	- 修改如下参数
	```bash
	Port 22 # 转发端口
	PermitRootLogin prohibit-password # 允许 root 用户登录
	AllowUsers <user_name> # 允许一般用户登录
	PasswordAuthentocation yes # 启用远程登录密码
	PermitUserEnvironment yes # 允许用户环境
	```
	- 重新启动 SSH 服务
	```bash
	sudo systemctl restart sshd
	sudo service ssh restart
	```
4. 在 VSCode 中建立 Remote-SSH 连接
	- 用户名为 WSL 的用户名
	- 密码为 sudo 密码