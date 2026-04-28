# Docker Deployment Notes

这个项目的 Docker 文件已经改成适合公开仓库使用的相对路径方案，不再依赖固定的本机目录。

## 目录约定

默认情况下，Compose 会使用这些目录：

- `./runtime/models`
- `./runtime/logs`
- `./runtime/shared`

如果你想改路径，复制 `.env.example` 为 `.env` 后修改对应变量即可。

## 快速开始

```bash
cp .env.example .env
mkdir -p runtime/models runtime/logs runtime/shared
./scripts/docker-build.sh
docker compose up -d rvc-server
```

健康检查：

```bash
curl http://127.0.0.1:8080/health
```

WebSocket 地址：

```text
ws://127.0.0.1:8080/rvc
```

## 常用环境变量

- `RVC_MODELS_DIR`：宿主机模型目录，映射到容器内 `/app/assets/weights`
- `RVC_LOGS_DIR`：日志目录
- `RVC_SHARED_DIR`：客户端共享目录
- `RVC_HOST_VENV`：`docker-compose.host-venv.yml` 使用的宿主机虚拟环境目录
- `RVC_SERVER_PORT`：`scripts/docker-run-server.sh` 使用的宿主机端口

## host-venv 模式

`docker-compose.host-venv.yml` 适合已经在宿主机上装好 Python、Torch 和 RVC 依赖的场景。使用前至少确认：

- `RVC_HOST_VENV` 指向可用虚拟环境
- `RVC_SOURCE_DIR` 指向当前仓库的 `src`
- `RVC_CONFIG_DIR` 指向当前仓库的 `configs`

示例：

```bash
docker compose -f docker-compose.host-venv.yml up -d rvc-server
```

## 注意

- 本仓库不包含模型权重
- 本仓库也不包含完整的上游 RVC 推理代码
- 如果服务端缺少 `infer.*` 或 `configs.config` 相关模块，容器能够启动到一定程度，但无法完成实际推理
