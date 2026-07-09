# Video Subtitle OCR

本机运行的视频字幕提取工具。启动一个 Python 服务，在浏览器里上传、拖入或粘贴视频，自动导出 `.srt` 和 `.txt`。

处理流程会优先检查视频里是否有可直接导出的文字软字幕；如果没有，再用 FFmpeg 抽帧、OpenCV 预处理、OCR 识别硬字幕。

## Features

- 单服务运行：FastAPI 后端和静态前端由同一个进程提供。
- 支持上传、拖拽导入、剪贴板粘贴导入视频。
- 支持在页面里拖动视频进度条后手动框选字幕区域。
- 支持停止正在运行的识别任务。
- 优先提取内嵌文字软字幕，避免不必要的逐帧 OCR。
- 使用 FFmpeg 抽帧，OpenCV 裁剪和增强字幕区域。
- 默认推荐 PaddleOCR CPU，代码层面兼容 OpenVINO、ONNXRuntime、EasyOCR 和 Tesseract。
- 自动配置抽帧 FPS、字幕裁剪区域、OCR batch size。
- 默认跨多帧投票锁定稳定字幕区域，减少车牌、Logo、画面文字等非字幕内容进入 OCR。
- 默认检测字幕区域变化，重复帧复用上一帧 OCR 结果，减少无效 OCR。
- 默认在自动裁剪区域内继续定位字幕细条，减少送入 OCR 的画面面积并降低背景运动干扰。
- 支持在页面中选择“中英混合”或“英文优先”字幕语言。
- 导出 SRT、TXT，并在页面内实时显示 OCR 样本日志。
- 可选配置 OpenAI、Anthropic、Ollama 或 OpenAI 兼容接口，在合并字幕后用大模型修正 OCR 拼写、空格和标点；支持页面测试连接、保存配置和一键重置。
- 提供健康检查、OCR 引擎检测 API，前端会自动禁用未安装的引擎。
- 默认在任务结束后清理原视频和抽帧，只保留输出文件、任务信息和日志。

## Platform Support

| 平台 | 支持状态 | 推荐入口 |
| --- | --- | --- |
| Windows 10/11 | 完整支持 | `start.bat` 或 `.\start.ps1` |
| macOS | 支持本机运行，OCR 依赖取决于上游 wheel | `bash start.sh` |
| Linux | 支持本机运行 | `bash start.sh` |
| Docker / Docker Compose | 推荐给跨平台部署和依赖隔离 | `docker compose up --build` |

如果 macOS 或某些 Linux 发行版原生安装 PaddlePaddle 失败，优先使用 Docker 运行。

## Quick Start

### Windows

```bat
start.bat
```

首次运行会自动创建 `.venv` 并安装推荐依赖。启动后打开：

```text
http://127.0.0.1:8000
```

PowerShell 也可以直接运行：

```powershell
.\start.ps1
```

### macOS / Linux

```bash
bash start.sh
```

如果想直接执行脚本，可以先加执行权限：

```bash
chmod +x start.sh scripts/install.sh
./start.sh
```

### Docker Compose

```bash
docker compose up --build
```

启动后打开：

```text
http://127.0.0.1:8000
```

停止服务：

```bash
docker compose down
```

任务文件、模型缓存和输出文件会保存在宿主机的 `./data` 目录。

`docker-compose.yml` 默认只绑定 `127.0.0.1:8000`，不会主动暴露到局域网。

如果你的 Docker 版本仍使用旧命令，也可以把 `docker compose` 换成 `docker-compose`。

## Smart Hybrid OCR

页面里的“智能选择”会按本机可用引擎自动决定策略：

- `快速预览`：优先使用 OpenVINO OCR / ONNXRuntime OCR，尽量减少复核，优先速度。
- `智能均衡` 和 `精细识别`：优先使用 OpenVINO OCR / ONNXRuntime OCR 做主识别；如果 PaddleOCR 也已安装，会自动对空结果、短噪声、疑似乱码、英文长串无空格等低可信帧做 PaddleOCR 复核。
- 如果没有可用加速后端，则自动回退到 PaddleOCR。

这种混合模式不是把所有帧都跑两遍，而是只复核可疑帧。它适合 Intel Ultra/Core 这类有核显或 NPU 的机器：尽量吃到 OpenVINO/ONNXRuntime 的速度，同时用 PaddleOCR 兜住准确率。

## Runtime Requirements

本机运行建议：

- Python 3.10 到 3.12
- CPU：普通多核 CPU 即可，默认 CPU OCR
- 内存：建议 16GB 以上，32GB 更舒服
- FFmpeg：可使用系统 FFmpeg；没有时会回退到 `imageio-ffmpeg`
- Tesseract：只有选择 Tesseract 引擎时才需要系统级可执行文件

一般占用取决于视频长度、抽帧 FPS 和 OCR 引擎。PaddleOCR CPU 首次加载模型时内存会明显上升，常见运行区间大约是 2GB 到 6GB；识别阶段会吃满部分 CPU 核心。服务关闭后不再占用 CPU/RAM，但 `.venv`、OCR 模型缓存和历史任务会继续占用磁盘空间。

## Install Profiles

项目支持多个安装档位：

| Profile | 内容 | 适合场景 |
| --- | --- | --- |
| `recommended` | Web 服务 + PaddleOCR CPU | 默认推荐 |
| `core` | Web 服务 + FFmpeg/OpenCV 基础处理，不含 OCR 引擎 | 只做软字幕提取或开发接口 |
| `openvino` | RapidOCR + OpenVINO | Intel CPU/iGPU/NPU 加速实验，适合 Ultra/Core 平台优先尝试 |
| `onnxruntime` | RapidOCR + ONNXRuntime | 跨平台高速 CPU 备选 |
| `easyocr` | EasyOCR | 试用备用 OCR，体积较大 |
| `tesseract` | pytesseract + 尝试安装系统 Tesseract | 轻量备用 OCR |
| `full` | PaddleOCR + OpenVINO + ONNXRuntime + EasyOCR + Tesseract | 想一次装齐所有引擎 |
| `dev` | 测试和 CI 所需轻量依赖 | 开发/CI |

默认一键启动只安装 `recommended`，避免把 OpenVINO、ONNXRuntime、EasyOCR/PyTorch 等大依赖一起拉下来。

## Windows

### One-click Start

```bat
start.bat
```

如果启动失败，窗口会停住并显示错误，不会直接闪退。常见原因是首次安装依赖失败、Python 没加入 PATH，或默认端口被占用；端口被占用时脚本会自动尝试后续端口。

指定端口：

```powershell
.\start.ps1 -Port 8010
```

不自动打开浏览器：

```powershell
.\start.ps1 -NoBrowser
```

只检查脚本路径和端口，不启动服务：

```powershell
.\start.ps1 -DryRun
```

### Install Engines

菜单式安装：

```bat
install.bat
```

PowerShell 安装指定档位：

```powershell
.\scripts\install.ps1 -Profile recommended
.\scripts\install.ps1 -Profile core
.\scripts\install.ps1 -Profile openvino
.\scripts\install.ps1 -Profile onnxruntime
.\scripts\install.ps1 -Profile easyocr
.\scripts\install.ps1 -Profile tesseract
.\scripts\install.ps1 -Profile full
.\scripts\install.ps1 -Profile dev
```

Tesseract 档位会优先检查本机是否已有 `tesseract.exe`。如果没有，会尝试通过 `winget` 安装 `UB-Mannheim.TesseractOCR`；如果你的系统没有 `winget`，就需要手动安装 Tesseract OCR，并确保它在 PATH 里。

## macOS

### Native Start

```bash
bash start.sh
```

指定端口：

```bash
bash start.sh --port 8010
```

不自动打开浏览器：

```bash
bash start.sh --no-browser
```

只检查脚本参数：

```bash
bash start.sh --dry-run
```

### Install Engines

```bash
bash scripts/install.sh --profile recommended
bash scripts/install.sh --profile core
bash scripts/install.sh --profile openvino
bash scripts/install.sh --profile onnxruntime
bash scripts/install.sh --profile easyocr
bash scripts/install.sh --profile tesseract
bash scripts/install.sh --profile full
bash scripts/install.sh --profile dev
```

如果需要 Tesseract，脚本会尝试使用 Homebrew：

```bash
brew install tesseract tesseract-lang
```

如果 PaddleOCR/PaddlePaddle 原生安装失败，建议使用 Docker Compose。

## Linux

### Native Start

```bash
bash start.sh
```

指定端口：

```bash
bash start.sh --port 8010
```

不自动打开浏览器：

```bash
bash start.sh --no-browser
```

### Install Engines

```bash
bash scripts/install.sh --profile recommended
bash scripts/install.sh --profile core
bash scripts/install.sh --profile openvino
bash scripts/install.sh --profile onnxruntime
bash scripts/install.sh --profile easyocr
bash scripts/install.sh --profile tesseract
bash scripts/install.sh --profile full
bash scripts/install.sh --profile dev
```

Tesseract 档位会尝试按系统包管理器安装：

- Debian/Ubuntu：`apt-get install tesseract-ocr tesseract-ocr-chi-sim`
- Fedora：`dnf install tesseract tesseract-langpack-chi_sim`
- Arch：`pacman -Sy tesseract tesseract-data-chi_sim`
- openSUSE：`zypper install tesseract-ocr tesseract-ocr-traineddata-chinese-simplified`

如果你的发行版不在列表里，请手动安装 Tesseract 并确保 `tesseract` 在 PATH 中。

## Docker

默认 Docker 镜像包含：

- FastAPI Web 服务
- FFmpeg
- PaddleOCR CPU
- pytesseract
- 系统 Tesseract + 简体中文语言包

启动：

```bash
docker compose up --build
```

后台启动：

```bash
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f
```

停止：

```bash
docker compose down
```

清理容器但保留本地 `./data`：

```bash
docker compose down --remove-orphans
```

构建包含 EasyOCR 的大镜像：

```bash
docker compose build --build-arg INSTALL_EASYOCR=true
docker compose up
```

或者直接用 Docker：

构建包含 OpenVINO / ONNXRuntime 的镜像：
```bash
docker compose build --build-arg INSTALL_OPENVINO=true --build-arg INSTALL_ONNXRUNTIME=true
docker compose up
```

Windows Command Prompt：

```bash
docker build -t video-subtitle-ocr .
docker run --rm -p 127.0.0.1:8000:8000 -e VSO_ALLOW_WEB_INSTALL=1 -v "%cd%/data:/app/data" video-subtitle-ocr
```

Windows PowerShell：

```powershell
docker build -t video-subtitle-ocr .
docker run --rm -p 127.0.0.1:8000:8000 -e VSO_ALLOW_WEB_INSTALL=1 -v "${PWD}/data:/app/data" video-subtitle-ocr
```

macOS/Linux：

```bash
docker run --rm -p 127.0.0.1:8000:8000 -e VSO_ALLOW_WEB_INSTALL=1 -v "$PWD/data:/app/data" video-subtitle-ocr
```

## Manual Install

Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

macOS/Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

只安装 Web 服务和基础处理能力：

```bash
python -m pip install -r requirements-core.txt
```

安装推荐 OCR：

```bash
python -m pip install -r requirements.txt
```

安装可选 OCR：

```bash
python -m pip install -r requirements-easyocr.txt
python -m pip install -r requirements-tesseract.txt
```

安装 OpenVINO / ONNXRuntime 高速后端：
```bash
python -m pip install -r requirements-openvino.txt
python -m pip install -r requirements-onnxruntime.txt
```

## PyCharm

1. 用 PyCharm 打开项目目录。
2. 选择 `.venv` 作为解释器；如果没有 `.venv`，先运行对应平台的一键启动或安装脚本。
3. 新建运行配置：
   - Module name: `uvicorn`
   - Parameters: `app.main:app --host 127.0.0.1 --port 8000`
   - Working directory: 项目根目录
4. 启动后访问 `http://127.0.0.1:8000`。

## OCR Engines

| 引擎 | 状态 | 说明 |
| --- | --- | --- |
| PaddleOCR | 默认推荐 | 中英文字幕综合效果最好，默认 CPU 运行。 |
| OpenVINO OCR | 可选高速后端 | 基于 RapidOCR/OpenVINO，适合 Intel CPU/iGPU/NPU 加速实验。速度优先，英文空格和复杂画面准确率可能不如 PaddleOCR。 |
| ONNXRuntime OCR | 可选高速后端 | 基于 RapidOCR/ONNXRuntime，跨平台 CPU 推理通常比纯 PaddleOCR 更轻。速度优先，准确率优先时仍建议 PaddleOCR。 |
| EasyOCR | 可选 | 代码已支持，但会安装 PyTorch，CPU 上通常更重。 |
| Tesseract | 可选 | 需要系统级 Tesseract OCR 程序和语言数据。 |

前端会调用 `/api/engines` 检测本机可用引擎，未安装的选项会自动禁用，并显示对应安装提示。

Web 页面里的“引擎安装”面板也支持跨平台：

- Windows：调用 `scripts/install.ps1`
- macOS/Linux/Docker：调用 `scripts/install.sh`

Web 安装器只允许预设安装档位，不接受自定义命令，并且只允许本机页面触发。

## Environment Variables

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VSO_MAX_UPLOAD_MB` | `4096` | 最大上传视频大小，单位 MB |
| `VSO_WORKERS` | 自动 | 强制指定后台识别 worker 数量。未设置时按 CPU 和可用内存自动规划 |
| `VSO_MAX_WORKERS` | 自动 | 自动规划 worker 的上限，适合用户手动限制最大并发 |
| `VSO_ESTIMATED_WORKER_GB` | `4.5` | 估算每个 OCR worker 需要的内存，自动并发规划会按它保留余量 |
| `VSO_CPU_THREADS` | 自动 | OpenCV/Paddle CPU 线程数上限 |
| `VSO_PADDLE_BATCH_SIZE` | `4` | PaddleOCR 批量大小 |
| `VSO_OPENVINO_DEVICE` | `AUTO` | OpenVINO 设备提示，可尝试 `AUTO`、`CPU`、`GPU`。具体是否可用取决于 OpenVINO 和本机驱动。 |
| `VSO_RAPIDOCR_MIN_SCORE` | `0.82` | OpenVINO/ONNXRuntime 后端的最低 OCR 置信度，调高会减少噪声但可能漏字。 |
| `VSO_CLEANUP_INTERMEDIATE_FILES` | `1` | 任务结束后是否删除原视频和抽帧 |
| `VSO_RETRY_FULL_FRAME_ON_EMPTY` | `0` | 空结果时是否尝试整帧 OCR |
| `VSO_ALLOW_WEB_INSTALL` | `0` | 是否允许非 loopback 客户端触发 Web 安装器。Docker 本机绑定时可设为 `1` |
| `PADDLE_PDX_CACHE_HOME` | `data/models/paddlex` | Paddle 模型缓存目录 |

Windows PowerShell 示例：

```powershell
$env:VSO_MAX_WORKERS="2"
$env:VSO_PADDLE_BATCH_SIZE="2"
.\start.ps1
```

macOS/Linux 示例：

```bash
VSO_MAX_WORKERS=2 VSO_PADDLE_BATCH_SIZE=2 bash start.sh
```

Docker Compose 示例：

```yaml
environment:
  VSO_MAX_WORKERS: "2"
  VSO_CPU_THREADS: "8"
  VSO_PADDLE_BATCH_SIZE: "4"
  VSO_OPENVINO_DEVICE: "AUTO"
```

## API

- `GET /api/health`：服务、队列和 FFmpeg 状态。
- `GET /api/engines`：OCR 引擎可用性。
- `GET /api/install/profiles`：安装档位列表。
- `GET /api/install/current`：当前安装任务状态。
- `POST /api/install/{profile}`：启动预设安装任务，仅允许本机页面触发。
- `POST /api/jobs`：上传视频并创建任务。
- `GET /api/jobs/{job_id}`：查询任务状态。
- `POST /api/jobs/{job_id}/cancel`：停止任务。
- `GET /api/jobs/{job_id}/ocr-log`：读取页面内 OCR 日志。
- `GET /api/jobs/{job_id}/download/srt`：下载 SRT。
- `GET /api/jobs/{job_id}/download/txt`：下载 TXT。
- `GET /api/jobs/{job_id}/download/preview`：下载 OCR 裁剪预览图。
- `GET /api/jobs/{job_id}/download/ocr-log`：下载 OCR 样本日志。

## Project Structure

```text
app/
  main.py              FastAPI 入口和 API
  processing.py        软字幕优先、抽帧、OCR、合并输出
  ocr_engines.py       OCR 引擎适配和可用性检测
  soft_subtitles.py    内嵌文字字幕检测和提取
  autotune.py          智能参数配置
  static/              前端页面
scripts/
  install.ps1          Windows/PowerShell 安装器
  install.sh           macOS/Linux 安装器
start.bat              Windows 一键启动
start.ps1              Windows/PowerShell 一键启动
start.sh               macOS/Linux 一键启动
Dockerfile             Docker 镜像
docker-compose.yml     Docker Compose 服务
data/                  本地任务、模型缓存和输出文件，默认不提交到 Git
tests/                 单元测试
```

## Development

安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt
```

校验：

```bash
python -m compileall app tests
python -m pytest
node --check app/static/app.js
```

Windows 下 JS 检查路径也可以写成：

```powershell
node --check app\static\app.js
```

CI 会在 Ubuntu、Windows、macOS 上跑轻量测试，避免跨平台脚本和核心代码退化。

## FAQ

**软字幕是不是不需要 OCR？**

是。只要视频里有文字软字幕轨道，服务会优先直接导出 SRT/TXT，不走逐帧 OCR。

**为什么第一次识别慢？**

首次运行需要加载 OCR 模型，可能还会下载或初始化缓存。之后会快一些。

**为什么 OCR 慢但 CPU 占用不高？**

PaddleOCR CPU 不是简单把所有核心打满的任务，它会在检测、方向分类、文字识别之间串行处理，并受 batch size、模型加载、磁盘读帧、文件系统性能影响。项目默认会自动设置 CPU 线程数和 PaddleOCR 批量大小；页面进度会显示 `frames/s` 和预计剩余时间。

如果进度长时间不变、Python CPU 也不增长，通常是旧服务或某次 OCR 调用卡住，建议关闭服务后重新启动。CPU 下 PaddleOCR 大 batch 偶尔会卡住，所以默认批量偏保守。可以用环境变量 `VSO_PADDLE_BATCH_SIZE` 调整，但如果出现进度不动、CPU 接近 0，应先降回默认值。

**批量和并发是一回事吗？**

不是。`VSO_PADDLE_BATCH_SIZE` / 页面里的“批量”表示 PaddleOCR 一次送入几张裁剪图；`VSO_WORKERS` 表示同时处理几个视频任务；`VSO_CPU_THREADS` 表示单个 worker 可用的 CPU 线程上限。默认不设置 `VSO_WORKERS` 时，服务会根据 CPU 核心数和启动时可用内存自动规划 worker 数，并受 `VSO_MAX_WORKERS` 限制。

**大模型纠错会不会必须配置？**

不会。默认完全不调用大模型。只有在页面里勾选“大模型纠错”并填写对应模型配置后，服务才会在 OCR 合并完成后调用 OpenAI、Anthropic、Ollama 或 OpenAI 兼容接口来修正字幕文本。它只修改字幕文字，不修改时间轴。页面提交的 API Key 只保存在当前任务内存里，不会写入 `job.json`，任务结束后随进程内任务对象释放。

**大模型配置能保存吗？**

可以。页面里的“大模型纠错（可选）”支持“测试连接”“保存配置”和“重置配置”。保存后配置会写入本机 `data/config.json`，下次打开页面会自动加载；API Key 不会回显到页面，但留空提交任务或测试连接时会继续使用已保存的 Key。`data/` 已在 `.gitignore` 中，发布到 GitHub 前不要手动提交这个私有配置文件。

**智能跳帧会不会影响字幕时间轴？**

默认不会丢时间点。服务仍然会给每个抽帧时间点生成样本；如果字幕区域和上一帧几乎一样，就复用上一帧 OCR 文本，合并字幕时仍然能延长同一句字幕的持续时间。页面 OCR 日志里带“复用”标记的行就是跳过 OCR 的重复帧。
当前实现会比较字幕区域的文字笔画掩膜，而不是直接比较整张画面，所以更能忽略背景轻微运动。自动模式下快速/均衡/精细的复用阈值会分别偏激进、均衡、保守；如果是纯英文视频，也建议在高级参数里选择“英文优先”。

**智能区域锁定是不是所有视频都不用手动框选？**

不是绝对的。服务会从多帧里投票寻找稳定字幕带，并把后续 OCR 固定在这个区域，能减少车牌、Logo、仪表盘文字、路牌等干扰；但遇到字幕位置频繁变化、花字标题、弹幕、竖屏拼接画面或字幕极小的素材时，手动拖到有字幕的时间点后框选仍然最稳。手动框选的优先级永远高于智能区域。

**竖屏视频为什么自动裁剪更窄？**

很多竖屏视频里实际内容是横向画面加上下黑边，字幕经常落在画面中下部。项目会对竖屏视频使用更紧的默认 OCR 区域，减少把车标、场景文字、界面字样一起送进 OCR 的概率。如果自动区域仍然不准，页面里拖到有字幕的时间点后手动框选字幕区域通常最稳。

**英文视频怎么更快？**

在“高级参数”里把“字幕语言”改成“英文优先”。这会让 PaddleOCR 使用英文模型，比默认“中英混合”更适合纯英文字幕。

**关闭窗口后还占资源吗？**

关闭服务后不再占用 CPU 和内存；`.venv`、`data/models`、`data/jobs` 会留在硬盘上。

Docker 下可以用 `docker compose down` 停止容器。`./data` 目录会保留，方便下次复用模型缓存和输出文件。

**识别后原视频会一直占磁盘吗？**

默认不会。任务完成、失败或停止后，会清理上传的原视频和抽出来的帧，只保留字幕文件、任务状态和 OCR 日志。如果你想保留中间文件，可以设置环境变量 `VSO_CLEANUP_INTERMEDIATE_FILES=0` 后再启动服务。

**识别不出来怎么办？**

先看页面内 OCR 日志。如果日志为空或识别到了画面里的杂字，建议在视频预览里拖到有字幕的位置，点击“框选区域”，只框住字幕所在区域再识别。

**Docker 里安装 EasyOCR 后重建容器会丢吗？**

如果是在 Web 页面里安装，依赖安装在当前容器层里，删除并重建容器后会丢。需要长期使用 EasyOCR 时，推荐用：

```bash
docker compose build --build-arg INSTALL_EASYOCR=true
docker compose up
```

## License

MIT License. See [LICENSE](LICENSE).
