# OpenSandbox、E2B、Daytona 技术预研与选型建议

> 调研日期：2026-08-21  
> 项目范围：`raster-agent` 的 Python / Shell 执行、地理栅格文件处理和多会话工作区隔离。

## 1. 执行摘要

三者不应再被简单视为三个同类“开源 Sandbox”：

- **OpenSandbox** 是当前最适合本项目走“开源、自托管优先”路线的候选。它提供 Python SDK、Docker/Kubernetes 后端以及可选的 gVisor、Kata、Firecracker 隔离；但默认仍是共享宿主内核的 runc，生产安全需要显式组合 Kata/Firecracker、出口控制、外部 IAM、审计和高可用存储。[^os-arch][^os-secure]
- **E2B** 默认以每个 Sandbox 一个 Firecracker microVM 作为隔离边界，托管接入简单，且 SDK、基础设施和 Dashboard 均为 Apache-2.0 开源。它是“托管优先、默认强隔离”的最佳基线；但完整自托管需要 KVM/Firecracker、Nomad/Consul、Postgres、Redis、对象存储和可观测体系，明显不是轻量单机方案。[^e2b-arch][^e2b-infra]
- **Daytona** 必须拆成两个方案评价：当前 Daytona Cloud/BYOC 是闭源托管产品；原 AGPL 服务端仓库在 2026 年 6 月后停止维护。它仍有成熟 SDK、生命周期和工作区体验，但已不适合作为新项目的持续维护开源自托管主线。[^daytona-close][^daytona-repo]

建议采用双轨验证：

1. **OpenSandbox 作为 self-host 主 PoC**，先用 Docker 验证接口和文件链路，再在 Linux/Kubernetes 上验证 Kata/Firecracker。
2. **E2B Cloud 作为 managed/安全基准 PoC**，验证相同镜像、数据、网络和生命周期测试。
3. **Daytona 仅在团队接受闭源控制面和供应商锁定时进入 Cloud/BYOC PoC**；不建议基于冻结的 OSS v0.190 新建生产平台。

## 2. 横向比较

| 维度 | OpenSandbox | E2B | Daytona Cloud/BYOC | Daytona OSS v0.190 |
|---|---|---|---|---|
| 当前定位 | 活跃开源编排与执行平台 | 活跃开源基础设施 + 托管云 | 闭源托管控制面 | 冻结的旧开源服务端 |
| 许可证 | Apache-2.0[^os-license] | Apache-2.0[^e2b-repo] | SDK Apache-2.0；核心闭源[^daytona-clients] | AGPL-3.0[^daytona-repo] |
| 默认隔离 | Docker/runc，共享宿主内核[^os-secure] | 每 Sandbox 一个 Firecracker microVM[^e2b-arch] | 默认 Linux 容器；VM 为单独类型[^daytona-arch][^daytona-sandbox] | 需按冻结版本逐项验证 |
| 最强隔离 | Kata + QEMU/Firecracker | Firecracker | VM Sandbox | 不应依赖后续 Cloud 文档推断 |
| 本地 PoC | 简单，Docker + Server | 托管简单；自托管困难 | 托管简单 | Compose 组件较多 |
| 生产自托管 | Kubernetes 组合式部署 | 完整但运维重 | BYOC 不等于开源自托管控制面 | 不推荐，已停止维护 |
| 网络控制 | 可选 egress sidecar、FQDN 策略和 Credential Vault[^os-egress] | netns + nftables；SDK 默认允许联网[^e2b-arch][^e2b-sdk] | Cloud 支持域名/CIDR/全禁策略[^daytona-network] | 需对精确 tag 验证 |
| 凭据保护 | 可在 Sandbox 外保存并按目标注入[^os-egress] | 注入的环境变量可被 guest 读取[^e2b-env] | Cloud 支持代理替换 opaque token[^daytona-secrets] | 需对精确 tag 验证 |
| 生命周期 | TTL、renew、pause；快照语义因 Docker/K8s 后端而异 | create/connect/kill、模板、pause/resume Beta、Volume | Cloud 的 VM/容器持久化能力丰富 | 功能冻结 |
| 多租户/IAM | 基础 API key/token；平台级 RBAC、审计需外补 | team/OIDC/admin 等控制面能力更完整 | Cloud 有组织与 scoped key | 旧 Dex/OIDC，停止演进 |
| `raster-agent` 初步结论 | **首选自托管 PoC** | **首选托管与强隔离基准** | 有条件的商业候选 | **不建议新采用** |

## 3. 架构与隔离分析

### 3.1 OpenSandbox

OpenSandbox 的主要层次是 SDK/CLI/MCP、FastAPI 生命周期服务、Docker 或 Kubernetes runtime、Sandbox 内的 `execd`，以及可选的 ingress/egress 组件。`execd` 提供命令流、后台进程、PTY、文件操作、Jupyter 执行和本地指标。[^os-arch][^os-execd]

其最大优点是组合灵活：本地可先使用 Docker，生产可以转 Kubernetes，并通过 RuntimeClass 使用 gVisor 或 Kata。最大风险同样来自这种组合性：默认 secure runtime 为空，即普通 runc；OpenSandbox 只校验已配置的安全 runtime，并不替运维方安装或保证其安全配置。[^os-secure]

对于不可信代码，建议生产基线使用 Kata VM 边界。gVisor 可作为风险较低、偏性能场景的折中，但它和 OpenSandbox 的 egress sidecar 存在兼容限制；需要出口策略时应采用 Kata，或交由 CNI 层实施经过验证的 FQDN 策略。[^os-secure][^os-egress]

### 3.2 E2B

E2B 把控制面与数据面分离：API 负责认证、配额、调度和生命周期，Redis 保存运行态路由，Postgres 保存模板、构建、快照和 Volume 等持久实体；数据面节点以 Firecracker 运行 microVM，guest 内的 `envd` 提供命令、PTY 和文件接口。[^e2b-arch]

模板本质上包含预启动 VM 的内存、磁盘与 VM 状态；启动时使用按需加载和 COW overlay，暂停时保存 dirty-block 差异。该设计适合快速创建强隔离环境，但自托管涉及 KVM 节点、root 权限 orchestrator、调度、数据库、对象存储和日志指标系统。官方仓库的支持矩阵应优先于营销表述；当前不应把通用 Linux、Azure 或单机裸金属理解为开箱即用的社区部署路径。[^e2b-arch][^e2b-infra]

### 3.3 Daytona

Daytona 当前文档描述 interface、control、compute 三个 plane，包含多语言 SDK、API、proxy、runner、数据库、对象存储和 Sandbox 内 daemon。[^daytona-arch] 但其安全文档存在关键表述冲突：“每个 Sandbox 有 dedicated kernel”与“默认使用 Linux container”不能同时成立。架构文档明确默认容器依赖 Linux namespaces，因此只有 VM Sandbox 才能合理视为独立 guest-kernel 边界。[^daytona-arch][^daytona-sandbox]

更重要的是，Daytona 已宣布核心生产代码转为闭源，原公开仓库不再维护。当前 Cloud 文档中的 VM、密钥代理、可观测性和持久化能力，不能未经验证就归属于冻结的旧 OSS tag。[^daytona-close]

## 4. 与 `raster-agent` 的代码适配

项目已经具备合适的接入边界：[`Sandbox`](./src/agent_core/sandbox/sandbox.py) 定义命令与文件操作，[`SandboxProvider`](./src/agent_core/sandbox/sandbox_provider.py) 负责按会话获取和释放实例。因此三方适配器应放在 provider/sandbox 层，而不是直接改写每个 Tool。

当前实现存在四个必须先解决的共性问题：

1. **文件接口仅支持文本。** GeoTIFF、GeoPackage、ZIP、PNG 等不能通过现有 `read_file`、`write_file` 和 `save_output_file` 安全传输。
2. **缺少产物发布通道。** 当前容器主要挂载 `workspace/`，而 HTTP 下载从本地 `outputs/` 返回；远程 Sandbox 生成的文件必须流式下载或复制到受控 artifact store。
3. **生命周期语义不匹配。** Tool 热路径会频繁 `acquire/release`。远程平台应按 `(user_id, conversation_id)` 复用租约，不能把每次 Tool 调用映射为创建/销毁一个 Sandbox。
4. **缺少地理栅格运行镜像和上传策略。** 默认 `python:3.11-slim` 没有 GDAL/PROJ/rasterio；上传白名单也未覆盖 `.tif/.tiff` 等格式。三种产品都需要同一个固定版本的 Linux geospatial 镜像。

建议先扩展中立接口：

```python
class Sandbox:
    def upload_file(self, local_path: Path, remote_path: str) -> None: ...
    def download_file(self, remote_path: str, local_path: Path) -> None: ...
    def publish_artifact(self, remote_path: str, output_name: str) -> Artifact: ...
    def stat(self, path: str) -> FileStat: ...
```

所有 provider 使用同一套 conformance tests，验证命令参数、stdin/env、超时、中止、退出码、二进制校验和、路径穿越、网络拒绝、会话复用和 TTL 清理。

## 5. 安全生产基线

对运行不可信 Python/Shell 的 `raster-agent`，建议最低要求如下：

- 每任务或每受控会话独立 microVM/VM：E2B Firecracker，或 OpenSandbox Kata/Firecracker；Daytona 仅接受明确的 VM class。
- 默认拒绝 ingress 和 egress，只放行精确的 HTTPS 目标；额外验证 DNS、IPv6、raw IP、重定向、私网地址和云 metadata 地址。
- 不把长期 LLM/API 密钥放入 guest 环境变量、文件、镜像或快照；使用目标绑定的凭据代理或外部短期 token broker。
- 禁止 privileged、Docker socket、hostPath、宿主项目目录和跨租户共享 RW volume；输入只读，工作区独占且临时，输出经带路径校验的 copy-out 服务发布。
- 在 guest 外强制 CPU、内存、PID、磁盘、inode、执行时间、创建频率和租户总配额。
- 固定镜像和组件 digest，验证签名/来源；记录生命周期、命令、网络和产物审计事件，并持续运行逃逸、越权、路径穿越、快照泄漏与清理测试。

## 6. PoC 方案与通过条件

### 阶段 A：公共能力改造

1. 增加二进制 upload/download 与 `publish_artifact`。
2. 把 provider 生命周期改成会话 lease；明确 stop、pause、archive、destroy 语义。
3. 制作一个固定版本的 Linux GDAL/PROJ/rasterio 镜像。
4. 建立 provider-neutral conformance tests。

### 阶段 B：OpenSandbox PoC

先在 Linux/WSL2 Docker 验证 SDK、镜像、工作区同步和失败清理；随后在 Kubernetes + Kata/Firecracker 上复测隔离、出口策略、快照、并发和节点故障。Docker PoC 只验证功能，不作为不可信多租户安全结论。

### 阶段 C：E2B Cloud 对照 PoC

使用相同镜像和测试数据，验证：自定义模板、禁止联网与域名 allowlist、pause/resume、Volume 持久性、文件吞吐、SDK 超时/取消和费用。所有容量、延迟和成本数据只记录实测值，不使用营销数字。

### 阶段 D：可选 Daytona Cloud/BYOC PoC

仅在商业条件允许时执行；合同或技术验收中明确 VM 隔离类型、数据驻留、退出导出、版本兼容、SLA、安全通告和 BYOC 控制面边界。

统一功能验收任务：上传小型 GeoTIFF → 执行 `gdalinfo` 和 rasterio 处理 → 生成派生 GeoTIFF 与 PNG → 发布并下载 → 校验哈希 → 重新获取同一会话并验证预期持久性 → 到期后确认彻底清理。

## 7. 最终建议

当前推荐排序取决于组织目标：

- **开源、自托管和可控性优先：OpenSandbox。** 它和现有 provider 抽象最匹配，也有从 Docker 到 Kubernetes 的渐进路径；立项时必须同时预算 IAM、审计、Kata、控制面 HA、备份和版本治理。
- **快速上线、默认强隔离、托管优先：E2B。** 它应作为安全和开发体验基准；只有具备成熟平台团队时才考虑完整自托管。
- **接受闭源、重点采购 managed/BYOC：Daytona Cloud。** 可以进入商务与技术验证，但需要供应商退出方案。
- **Daytona OSS v0.190：不建议新采用。** 停止维护和 AGPL 服务端带来的安全、升级与合规成本超过其短期功能收益。

## 8. 信心评估

- **高信心：** 三者默认隔离边界；OpenSandbox/E2B 的开源许可证和公开架构；Daytona 停止维护 OSS 核心的状态；本项目文本文件接口和 provider 生命周期问题。这些结论来自官方仓库、官方公告或本地代码。
- **中信心：** OpenSandbox 的生产 HA 完备度、各产品快照在故障场景下的行为。公开文档揭示了架构，但仍需用精确版本和实际集群验证。
- **待实测：** GDAL 镜像构建时间、GeoTIFF 吞吐、并发容量、冷启动/恢复延迟、Volume 语义、完整成本和网络绕过面。报告未对这些指标作估算。

## 参考资料

[^os-arch]: [OpenSandbox Architecture](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/architecture.md)
[^os-secure]: [OpenSandbox Secure Container Runtime](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/guides/secure-container.md)
[^os-execd]: [OpenSandbox execd](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/components/execd.md)
[^os-egress]: [OpenSandbox Egress and Credential Vault](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/components/egress.md)
[^os-license]: [OpenSandbox License](https://github.com/opensandbox-group/OpenSandbox/blob/main/LICENSE)
[^e2b-arch]: [E2B Infra Architecture](https://github.com/e2b-dev/infra/blob/main/docs/ARCHITECTURE.md)
[^e2b-infra]: [E2B Infra README and support matrix](https://github.com/e2b-dev/infra/blob/main/README.md)
[^e2b-repo]: [E2B SDK repository](https://github.com/e2b-dev/e2b)
[^e2b-sdk]: [E2B Python Sandbox SDK](https://e2b.dev/docs/sdk-reference/python-sdk/v2.14.0/sandbox_async)
[^e2b-env]: [E2B envd environment-variable API](https://e2b.dev/docs/api-reference/envd/get-the-environment-variables)
[^daytona-close]: [Daytona is Going Closed Source, 2026-06-11](https://www.daytona.io/dotfiles/updates/daytona-is-going-closed-source)
[^daytona-repo]: [Daytona public repository](https://github.com/daytonaio/daytona)
[^daytona-clients]: [Daytona client repositories and Apache-2.0 relicensing](https://www.daytona.io/changelog/client-repo-restructure-and-apache-20-relicense)
[^daytona-arch]: [Daytona Architecture](https://www.daytona.io/docs/en/architecture/)
[^daytona-sandbox]: [Daytona Sandboxes](https://www.daytona.io/docs/en/sandboxes/)
[^daytona-network]: [Daytona Network Limits](https://www.daytona.io/docs/en/network-limits/)
[^daytona-secrets]: [Daytona Secrets](https://www.daytona.io/docs/en/secrets/)
