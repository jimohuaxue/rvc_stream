# RVC Stream

`RVC Stream` 是一个基于 WebSocket 的实时语音转换项目，包含：

- 本地采集和播放音频的客户端
- 提供 WebSocket 接口的服务端
- 一个用于查看状态、调整参数和上传模型的管理面板

这个仓库适合公开分享“流式封装层”和部署脚本，但它不是一个开箱即用的完整 RVC 推理仓库。真正的推理仍依赖外部 RVC 运行时和模型文件。

## 项目定位

服务端代码会按需导入以下外部模块：

- `configs.config`
- `infer.lib.rtrvc`
- `infer.lib.jit.get_synthesizer`
- `infer.lib.infer_pack.models`

这些模块不在当前仓库内，所以公开上传时我保留了流式通信、配置管理、管理面板、测试、Docker 文件和安装脚本，但没有附带完整 RVC 推理代码、模型权重或运行日志。

## 当前仓库包含什么

```text
.
├── src/
│   ├── rvc_client.py              # WebSocket 客户端
│   ├── rvc_server.py              # WebSocket 服务端
│   └── admin/                     # 管理面板与配置接口
├── configs/server.yaml            # 默认服务配置
├── scripts/                       # Docker 构建和启动脚本
├── docker-compose.yml             # 通用 Docker Compose
├── docker-compose.host-venv.yml   # 复用宿主机虚拟环境的 Compose
├── setup_env.sh                   # Linux/macOS 环境安装脚本
├── setup_env.bat                  # Windows Conda 安装脚本
└── tests/                         # 基础测试
```

## 已做的公开整理

- 删除了测试缓存、`egg-info`、运行目录和内部交付/验收报告
- 去掉了原来写死的本机绝对路径
- 新增了 `.gitignore`，避免把模型、日志、虚拟环境和本地配置提交到 GitHub
- 新增了 `.env.example`，把 Docker 目录映射改为环境变量或相对路径

## 环境要求

- Python 3.10+
- 客户端需要可用的音频输入/输出设备
- 服务端建议使用 Linux + NVIDIA GPU
- 如果要真正做 RVC 推理，需要你自己准备完整的 RVC 依赖和模型文件

## 安装方式

### 1. Conda

```bash
conda env create -f environment.yml
conda activate rvcstream
python scripts/install_torch.py --variant cu124
```

### 2. pip

```bash
pip install -r requirements.txt
pip install -r requirements-client.txt
pip install -r requirements-server.txt
pip install -r requirements-admin.txt
pip install -r requirements-dev.txt
```

如果服务端机器需要完整 RVC 推理相关依赖，再额外执行：

```bash
pip install -r requirements-rvc.txt
python scripts/install_torch.py --variant cu124
```

## 配置说明

默认配置文件是 [`configs/server.yaml`](./configs/server.yaml)：

```yaml
server:
  host: 0.0.0.0
  port: 9000

streaming:
  sample_rate: 16000
  chunk_size: 512
  channels: 1

rvc:
  f0method: fcpe
  pitch: 5
  index_rate: 0.8
```

主要配置项：

- `server.host` / `server.port`：服务监听地址和端口
- `streaming.*`：采样率、chunk 大小、声道等流式参数
- `rvc.*`：RVC 推理参数，管理面板也会读写这部分
- `models`：模型列表，由管理面板持久化

## 本地运行

### 启动服务端

只启动 WebSocket 服务端：

```bash
python -m src.rvc_server --host 0.0.0.0 --port 8080
```

启动带管理面板的服务端：

```bash
python start_server_with_admin.py --host 0.0.0.0 --port 8080
```

管理面板默认挂载在：

```text
http://127.0.0.1:8080/admin
```

### 查看客户端音频设备

```bash
python -m src.rvc_client --list-devices
```

### 启动客户端

```bash
python -m src.rvc_client \
  --server ws://127.0.0.1:8080/rvc \
  --model your_model.pth \
  --index your_model.index
```

如果模型已经放在服务端的模型目录里，可以把 `--model` / `--index` 写成服务端可见路径。

## Docker 使用方式

### 1. 准备环境变量文件

```bash
cp .env.example .env
mkdir -p runtime/models runtime/logs runtime/shared
```

### 2. 构建镜像

```bash
./scripts/docker-build.sh
```

### 3. 启动服务端

```bash
docker compose up -d rvc-server
```

### 4. 查看健康检查

```bash
curl http://127.0.0.1:8080/health
```

如果你想复用宿主机已经装好的虚拟环境，可以使用：

```bash
docker compose -f docker-compose.host-venv.yml up -d rvc-server
```

这个模式要求你自己把 `.env` 中的 `RVC_HOST_VENV`、`RVC_SOURCE_DIR`、`RVC_CONFIG_DIR` 指到正确目录。

## 测试

```bash
python -m pytest tests -q
```

这些测试主要覆盖打包、基础接口和管理面板支持，不会替你验证完整的 RVC 推理链路。

## 上传到 GitHub 前建议再确认

- 模型权重、索引文件、日志和 `.env` 不要提交
- 如果你的本地 RVC 推理代码是另外复制进来的，也不要一起混到这个仓库里
- `pyproject.toml` 里目前声明的是 MIT，但仓库里还没有单独的 `LICENSE` 文件；公开前最好补上
- 如果你后续加了私有服务器地址、内网域名或认证信息，记得再次检查再推送
