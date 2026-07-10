# Codex Handoff Summary

## 1. 当前项目目标

项目名称：PCR-Net Temperature Downscaling。

项目用途：构建一个用于学术论文模型公开的开源仓库，整合数据获取与处理、物理基底生成、CatBoost teacher inference、PCR-Net 主体模型训练测试、基线模型训练测试、消融实验和可视化复现流程。

当前开发阶段：基础功能已完成，后续工作转为文档完善与用户测试维护。项目当前应被视为“可运行但仍需用户测试验证”的状态。Demo 和 Release 下载链路已经通过本地验证，但开源用户环境、网络环境、Windows 路径、Conda 配置、GPU/依赖组合仍可能暴露问题。

## 2. 当前代码状态

已完成的主要功能：

- 标准开源仓库结构已搭建：`src/`、`configs/`、`docs/`、`examples/`、`scripts/`、`assets/`、`data/`、`outputs/`、`runs/`。
- 旧 `data_part` 与 `model_part` 功能已迁移到正式源码目录：`src/data_pipeline`、`src/physical_base`、`src/open_source`、`src/pcr_downscaling`。
- Demo1 已能生成完整 model-ready 数据集格式，与真实 demo 数据集布局一致。
- Demo2 已实现 LST CatBoost 特征构建、CatBoost 训练、CatBoost inference、PCR-Net 训练、PCR-Net 测试。
- Demo3 已实现使用预训练权重或 Demo2 checkpoint 出图。
- Demo4 已扩展为核心消融矩阵：`no_attention`、`no_gradient_loss`。
- Demo5 已实现 RF、无 LST CatBoost、Basic U-Net 基线训练与测试。
- Demo2/3/4/5 支持 `--dataset smoke_case|mini_case` 和 `--version time|spatial`。
- Demo2/3/4/5 支持 Release artifact 检查/下载；`mini_case` 下载失败时回退到 `smoke_case`，如果 prepared dataset 都不可用则回退到 Demo1 的 `model_ready`。
- Demo3 会按 time/spatial 检查或下载对应预训练权重。
- Release artifact manifest 已建立：`configs/artifacts/artifact_manifest.json`。
- Release 下载脚本已建立：`scripts/download_release_artifact.py`。

关键目录和文件：

- 根 README：`README.md`
- 交接文档：`HANDOFF.md`
- 项目配置：`pyproject.toml`
- 兼容旧训练入口配置：`config.py`
- Artifact manifest：`configs/artifacts/artifact_manifest.json`
- 数据处理配置：`configs/data/`
- 物理基底配置：`configs/physical/`
- Demo 配置：`configs/demos/`
- 模型配置：`configs/model/`
- 数据契约文档：`docs/data_contract.md`
- 模型工作流文档：`docs/model_workflows.md`
- 手动下载说明：`docs/manual_downloads.md`
- Demo 数据说明：`assets/demo_data/README.md`
- 预训练权重说明：`assets/pretrained/pcr_net/README.md`

当前项目入口：

- 数据处理 Demo：`examples/01_data_fetch/run_demo.py`
- 主模型训练测试 Demo：`examples/02_train_and_test/run_demo.py`
- 可视化 Demo：`examples/03_visualization/run_demo.py`
- 消融实验 Demo：`examples/04_ablation_training/run_demo.py`
- 基线模型 Demo：`examples/05_baseline_training_and_test/run_demo.py`
- Release artifact 下载：`scripts/download_release_artifact.py`
- 数据流水线入口：`scripts/run_pipeline.py`
- 物理基底入口：`scripts/run_physical.py`

示例数据和权重位置：

- Git 内置 smoke 数据：`assets/demo_data/smoke_case/`
- Release 管理 mini 数据：`assets/demo_data/mini_case/`
- Release 数据归档：`assets/demo_data/pcrnet-mini-case-v0.1.0.tar.gz`
- Release 预训练权重：`assets/pretrained/pcr_net/pcr-time-v0.1.0.pth`、`assets/pretrained/pcr_net/pcr-spatial-v0.1.0.pth`
- Demo3 当前运行读取的权重别名：`assets/pretrained/pcr_net/pcr-time.pth`、`assets/pretrained/pcr_net/pcr-spatial.pth`

核心代码保护范围：以下文件和目录属于核心逻辑，不应在没有明确需求时大规模重构：

- `src/data_pipeline/**`
- `src/physical_base/**`
- `src/open_source/**`
- `src/open_source/PCR-Net/**`
- `src/open_source/CatBoost/**`
- `src/open_source/base_line/**`
- `src/open_source/ablation/**`
- `src/open_source/unet_training.py`
- `src/open_source/unet_datasets.py`
- `src/open_source/test/evaluate.py`
- `src/open_source/mapping/visualize.py`

`src/pcr_downscaling/**` 是为开源包装、demo、artifact、layout、fixture 等新增的仓库级辅助层，可以按需求小范围维护，但也应避免破坏现有 demo 契约。

## 3. 运行环境

本项目使用 Conda 管理 Python 环境。不要直接运行 `python`、`pip`、`pytest` 或其他 Python 相关命令，也不要假设系统 PATH 中存在可用 Python。

Conda 路径：

```powershell
D:\MiniConda\Scripts\conda.exe
```

所有 Python 命令必须使用：

```powershell
& "D:\MiniConda\Scripts\conda.exe" run -n <env_name> python ...
```

环境用途：

- `geo_env`：数据处理、栅格/矢量、HDF5/NPY/CSV、GeoTIFF、NetCDF、坐标转换、特征生成。
- `plot_env`：纯绘图任务。注意：当前可视化 demo 已按用户要求使用 `torch_gpu` 测试。
- `torch_gpu`：PyTorch 训练、推理、测试、PCR-Net、U-Net、CatBoost teacher 后模型链路、Demo2/3/4/5。

深度学习任务前建议先检查 GPU：

```powershell
& "D:\MiniConda\Scripts\conda.exe" run -n torch_gpu python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Conda 临时文件注意事项：

- 不要并行运行多个 `conda run`，此前并行测试触发过 Windows 临时文件占用。
- 建议每次测试设置独立临时目录，例如：

```powershell
$tmp = Join-Path (Get-Location) 'temp\conda_run_xxx'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:TEMP = $tmp
$env:TMP = $tmp
& "D:\MiniConda\Scripts\conda.exe" run -n torch_gpu python ...
```

禁止事项：

- 不要删除或重建整个 Conda 环境；
- 不要执行 `conda clean --all`；
- 不要删除用户目录下的 `.conda`、`pkgs` 或 `envs`；
- 不要修改系统 PATH；
- 不要结束不相关进程；
- 不要使用管理员权限；
- 不要删除已有模型权重和 demo 数据。

## 4. 当前验证情况

已通过的主要验证：

- Demo1：`geo_env` 下运行 `examples/01_data_fetch/run_demo.py`，通过。
- Demo1 生成完整 `outputs/demos/01_data_fetch/model_ready/` 数据集布局，包括 `manifest.json`、`validation_summary.json`、`model_inputs/splits/*.csv`、ERA5/LST/tbase HDF5、`static.npy`、`truth.npy`。
- Demo1 数据格式检查：`dataset=demo1_model_ready`，`years=[2008, 2009, 2010]`，`sample_universe_count=12`，`truth=(1, 1096, 3)`，ERA5/LST/tbase 三年文件 shape 正确。
- Demo2 使用 Demo1 数据运行 `--data-root outputs/demos/01_data_fetch/model_ready --version time`，通过。
- Demo2 会把 CatBoost inference 写回 Demo1 数据目录：`outputs/demos/01_data_fetch/model_ready/catboost_inference/catboost_lst/`。
- Demo4 使用 Demo1 数据运行 `--data-root outputs/demos/01_data_fetch/model_ready --version time`，通过。
- Demo2 `--dataset mini_case --version time --smoke-only`，通过。
- Demo3 `--dataset mini_case --version time`，通过并出图。
- Demo4 `--dataset mini_case --version spatial --smoke-only`，通过。
- Demo5 `--dataset mini_case --version time --smoke-only`，通过。

Release artifact 下载脚本验证：

```powershell
& "D:\MiniConda\Scripts\conda.exe" run -n torch_gpu python scripts/download_release_artifact.py mini_case
& "D:\MiniConda\Scripts\conda.exe" run -n torch_gpu python scripts/download_release_artifact.py pcr_time
& "D:\MiniConda\Scripts\conda.exe" run -n torch_gpu python scripts/download_release_artifact.py pcr_spatial
```

结果：已有资产路径通过。

真实 Release 下载测试：

- 使用隔离 manifest 下载到 `temp/release_download_probe_*`，未删除或移动真实 `assets/`。
- `pcrnet-mini-case-v0.1.0.tar.gz` 下载并解压通过。
- `pcr-time-v0.1.0.pth` 下载并安装为 `pcr-time.pth` 通过。
- `pcr-spatial-v0.1.0.pth` 下载并安装为 `pcr-spatial.pth` 通过。

下载失败回退测试：

- 使用坏 URL 隔离 manifest 模拟 3 次网络失败。
- Demo2/3/4/5 请求 `mini_case` 后均回退到 `smoke_case`，没有直接崩溃。
- Summary 中记录 `requested_dataset: mini_case`、`dataset: smoke_case` 和下载失败输出。

尚未完整运行的测试：

- 没有在全新 clone 环境从零跑完整文档式 Quick Start。
- 没有在 Linux/macOS 上测试路径和权限。
- 没有重新运行完整 `mini_case` 的 Demo2/4/5 全训练矩阵，只做了关键路径和 smoke 验证。
- 没有运行完整数据下载链路的 ERA5/MODIS/Landsat/CCI/GEE 全模块验证。
- 没有执行单元测试框架，因为当前项目尚未建立正式 pytest 测试套件。

## 5. 已知问题和风险

截至本交接，未保留明确的已知代码 Bug。仍需用户实际测试验证。

平台和环境风险：

- Windows + Conda `conda run` 并行时可能出现临时文件占用。请顺序执行，并设置独立 `TEMP/TMP`。
- Windows 路径中反斜杠、中文用户名、权限继承可能影响临时目录、tar 解压或 `.part` 文件重命名。
- `scripts/download_release_artifact.py` 已对 `.part` 重命名失败做兼容：如果 `.part` 通过 hash/size 校验，可直接用于安装。
- Codex sandbox 可能限制外部盘、网络、系统临时目录或某些文件操作。
- GitHub Release 网络访问可能失败，demo 应回退到 `smoke_case` 或 Demo1 数据，但真实用户环境仍需文档说明。

不应修改或删除的大文件/数据：

- `assets/demo_data/mini_case/`
- `assets/demo_data/pcrnet-mini-case-v0.1.0.tar.gz`
- `assets/pretrained/pcr_net/pcr-time-v0.1.0.pth`
- `assets/pretrained/pcr_net/pcr-spatial-v0.1.0.pth`
- `assets/pretrained/pcr_net/pcr-time.pth`
- `assets/pretrained/pcr_net/pcr-spatial.pth`
- `outputs/demos/01_data_fetch/model_ready/`，除非明确需要重新生成 Demo1 数据。
- 用户外部真实研究数据路径，如 `E:\DataSet\...` 或 `F:\DataSet\...`。

用户测试最可能出问题的环节：

- Conda 环境缺依赖或环境名不一致。
- GPU/CUDA/torch 版本差异。
- Release 下载网络失败、代理设置、GitHub 访问受限。
- Windows 权限导致临时文件、`.part` 文件、tar 解压异常。
- Demo3 预训练权重命名或下载状态不一致。
- Demo1 不是完整真实数据下载链路，只是最小数据处理 + model-ready 数据契约验证。
- README 仍需统一更新，避免用户误以为 Demo1 可下载所有真实研究数据。

## 6. 后续任务清单

文档完善优先级：

1. 更新根 `README.md`：项目简介、环境准备、Release artifact 下载说明、smoke/mini/Demo1 fallback 数据区别、一条命令运行 Demo1-5、时间/空间泛化命令、常见错误处理。
2. 完善 Quick Start：最小 smoke 流程、使用 Release `mini_case`、使用预训练权重出图、从 Demo1 fallback 开始跑 Demo2/4。
3. 完善数据准备文档：`docs/data_contract.md`、`docs/manual_downloads.md`、原始数据、手动下载、GEE、代理说明。
4. 完善配置说明：`configs/artifacts/artifact_manifest.json`、`configs/demos/data_fetch_minimal.yaml`、`config.py`。
5. 完善各 demo README：输入、输出、环境、失败回退、与论文复现关系。
6. 增加 FAQ：Conda 找不到、CUDA 不可用、GitHub Release 下载失败、Windows 权限问题、预训练权重不匹配、`mini_case` 和 `smoke_case` 区别。

用户测试反馈后的 Bug 修复流程：

1. 先复现用户命令，严格使用指定 Conda 环境。
2. 若是网络/环境问题，先报告原因，不要擅自安装包或修改环境。
3. 若是代码问题，定位最小影响范围，优先修 demo/helper 层，不改核心模型逻辑。
4. 修复后运行对应最小 demo 或 smoke 验证。
5. 更新 README/FAQ 中相关问题。
6. 不自动 commit、reset 或 clean。

建议优先处理的 TODO：

- 将当前测试命令整理成 `README.md` 的 Quick Start。
- 补充 Release artifact manifest 字段解释和下载脚本用法。
- 明确 Demo1、smoke_case、mini_case、真实完整研究数据之间的关系。
- 添加“使用代理下载 Release/数据”的说明。
- 为 Windows Conda 临时文件问题增加 FAQ。
- 考虑增加一个只读校验脚本，用于检查 demo 数据集格式，不启动训练。

暂时不建议做的任务：

- 不建议大规模重构 `src/open_source`。
- 不建议替换核心训练架构、loss、dataset、evaluation 逻辑。
- 不建议重新设计目录结构。
- 不建议把 Release 管理的大文件提交进 git。
- 不建议在没有用户确认时重新切割真实 `mini_case` 数据集。
- 不建议自动清理 `outputs/`、`runs/` 或 `temp/`，除非用户明确要求。

## 7. 新 Codex 对话的启动指令

可复制到新 Codex 对话中的提示词：

```text
请接手 E:\pcr-net-temperature-downscaling 项目。

先不要修改任何代码。请先阅读项目根目录的 HANDOFF.md、TODO.md、AGENTS.md；如果 TODO.md 或 AGENTS.md 不存在，请说明缺失但继续。然后运行 git status，查看当前未提交/未跟踪文件状态。

阅读后请先总结：
1. 当前项目状态；
2. 当前可运行的 demo 和 Release artifact 状态；
3. 你认为下一步应优先做的文档或维护任务；
4. 如果需要修改文件，先给出计划并等待我确认。

注意：
- 所有 Python 命令必须通过 D:\MiniConda\Scripts\conda.exe run -n <env_name> python ... 执行；
- 不要直接运行 python、pip、pytest；
- 不要删除 Conda 环境、权重、demo 数据或用户外部数据；
- 不要自动 commit、reset、clean；
- 不要大规模重构核心代码。
```

## 8. Git 状态

最近一次执行：

```powershell
git status --short
```

结果摘要：

```text
?? .gitignore
?? HANDOFF.md
?? README.md
?? assets/
?? config.py
?? configs/
?? data/
?? docs/
?? examples/
?? outputs/
?? pyproject.toml
?? runs/
?? scripts/
?? src/
```

当前仓库看起来仍处于整体未跟踪状态，大部分文件都是 `??`。因此：

- 新对话修改前必须先查看 `git status`。
- 如果文件已被跟踪，再查看 `git diff`。
- 如果文件仍未跟踪，`git diff` 不会显示内容变化，需要直接读取文件。
- 不要自动提交。
- 不要自动 `git reset`。
- 不要自动 `git clean`。
