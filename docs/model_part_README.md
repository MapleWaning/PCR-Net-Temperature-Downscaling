# 模型训练根目录说明

本目录是气温降尺度开源项目中的模型训练部分，后续会与数据预处理项目合并。它已经从研究期工程中拆出并整理为较清晰的训练、测试和可视化代码树；合并时应把这里视为“模型训练根目录”，不要依赖旧工程中的历史脚本、生成产物或实验残留。

## 当前范围

当前代码覆盖三类模型工作流：

- 表格残差模型：CatBoost、无 LST CatBoost baseline、无 LST Random Forest baseline。
- 神经网络模型：PCR-Net、baseline U-Net、ResNet U-Net 消融实验。
- 神经网络测试和样本级可视化：统一精度指标评估，以及从测试集抽取 n 个样本生成对比图。

尚未包含完整神经网络全域 HDF5 推理保存流程，也未包含 demo、论文作图汇总脚本或研究期批量产物。

## 目录结构

```text
模型训练根目录/
  data_prepare/        表格模型特征构建与 schema
  CatBoost/            带 LST 特征的 CatBoost 残差模型入口
  base_line/
    CB/                无 LST CatBoost baseline
    RF/                无 LST Random Forest baseline
    U-Net/             baseline U-Net 训练入口
  PCR-Net/             PCR-Net 训练入口
  ablation/            PCR-Net / ResNet U-Net 消融训练入口
  test/                神经网络统一精度测试
  mapping/             神经网络样本级可视化
  *_training.py        共享训练工具
  *_inference.py       共享空间推理工具
  unet_*.py            神经网络数据集、模型、loss 和训练循环
```

各级 `__init__.py` 只用于把目录声明为 Python package，不承载业务逻辑。

## 表格模型部分

`data_prepare/` 负责为 CatBoost 和 Random Forest 构建 parquet 特征数据。特征定义集中在 schema 中：

- 带 LST 的 CatBoost 使用 ERA5、LST、DEM、land use、albedo、day-of-year sin/cos 特征。
- 无 LST 的 CB/RF baseline 使用 ERA5、DEM、land use、albedo、day-of-year sin/cos 特征。
- 残差目标为平均温、最高温、最低温三个通道。

数据构建入口只写新的 parquet：`train.parquet`、`val.parquet`、`test.parquet`。不要恢复旧的 `processed_data/*.npy` 流程。

表格模型训练复用共享 CatBoost helper 和空间推理 helper：

- 主 CatBoost 入口在 `CatBoost/`，用于带 LST 特征的残差训练和全域推理。
- `base_line/CB/` 是无 LST CatBoost baseline。
- `base_line/RF/` 是无 LST Random Forest baseline。
- 全域推理入口会读取模型和空间特征，输出残差预测结果；合并时需要与数据预处理项目约定好 parquet 字段名、空间网格字段和输出目录。

## 神经网络公共模块

神经网络代码的核心在四个共享模块：

- 数据集模块提供 PCR-Net 年份泛化、PCR-Net 站点泛化、baseline U-Net 年份泛化、baseline U-Net 站点泛化四种 dataset。
- 模型模块提供 `BasicUNet` 和 `ResAttUNet`。`ResAttUNet` 是当前 PCR-Net / 消融使用的 ResNet U-Net 结构。
- loss 模块提供 PCR-Net guided loss 和 baseline 纯 MSE loss。
- 训练模块提供 dataset 构建、batch 组装、训练循环和三类 fit 函数。

神经网络输入通道顺序固定为：

```text
base temperature, DEM features, albedo, time features, land-use one-hot
```

land use 会转换为 10 类 one-hot，因此默认输入通道数是 20。PCR-Net dataset 额外提供 tree-model guidance map，用于 guided loss；baseline dataset 不提供 guidance map。

## 神经网络训练入口

`PCR-Net/` 是主模型训练入口，默认使用 `ResAttUNet`、PCR-Net dataset 和 guided loss。它支持：

- `--generalization year`
- `--generalization station`
- `--use-attention` / `--no-use-attention`
- `--use-sft` / `--no-use-sft`
- `--no-pretrained-backbone`

`base_line/U-Net/` 是 baseline U-Net 训练入口，使用 `BasicUNet` 和纯 MSE loss，也支持 year/station 两种泛化设置。

`ablation/` 中的入口脚本只负责设置不同的训练参数，再调用公共训练函数。当前保留的消融包括：

- 关闭 attention 和 SFT。
- 关闭 attention gate。
- 空间泛化版本的无 attention / SFT。
- 去掉 gradient guidance loss，改用纯 MSE。
- 空间泛化版本的无 gradient guidance loss。
- 使用非 advanced physical base 的版本。

CBAM 是历史残留，当前模型训练根目录中故意不包含 CBAM 模块、`use_cbam` 参数或 no-CBAM 消融。合并时不要把旧工程中的 CBAM 代码重新带入。

## 神经网络测试

`test/` 是 PCR-Net、baseline U-Net 和消融实验共用的精度测试代码。它只保留四个精度指标：

```text
RMSE, MAE, MBE, R2
```

计算口径为：按 mask 取有效像素，先把预测和标签反归一化到摄氏度，再按平均温、最高温、最低温三个通道累计指标。MBE 使用 `prediction - target` 的平均偏差。

测试代码不迁移 TPI 分类、物理一致性、Pearson、Excel logging 或独立 MSE 输出。CSV 保存也只保存上述四个指标和有效像素数。

典型入口形式：

```powershell
python test\evaluate.py --dataset-type pcr-net --model-type resattunet --generalization year --model-path <checkpoint>
python test\evaluate.py --dataset-type baseline --model-type basic-unet --generalization station --model-path <checkpoint>
python test\evaluate.py --dataset-type pcr-net --model-type resattunet --no-use-attention --no-use-sft --model-path <checkpoint>
```

## 样本级可视化

`mapping/` 迁移的是旧可视化脚本中单样本对比图的核心逻辑，并改造成一次输出 n 张图。它会加载 dataset 和模型权重，抽取样本后生成四联图：

- Terrain Elevation (DEM)
- Input Base Temperature
- Model Refined Temperature
- Correction (Prediction - Base)

所有图像标题、colorbar、日志和文件名均使用英文。默认随机抽样，也可以通过 sample index、station id 或 date 过滤样本。

典型入口形式：

```powershell
python mapping\visualize.py --n-samples 10 --model-path <checkpoint>
python mapping\visualize.py --sample-indices 0,10,20 --model-path <checkpoint>
python mapping\visualize.py --station-id <station> --date 2023-08-06 --n-samples 1 --model-path <checkpoint>
```

这部分不是完整全域推理管线，只是面向模型结果检查的样本级可视化。

## 配置和数据依赖

当前入口脚本默认从项目根目录的 `config.py` 读取路径和超参数，包括：

- 训练、验证、测试年份。
- station split 的 metadata CSV。
- 静态数据、truth label、base temperature、guidance map 路径。
- batch size、worker 数、输入/输出通道数。
- 默认模型权重路径和可视化样本数。

合并到完整项目时，建议保留 CLI 参数覆盖能力，不要把数据预处理项目的路径硬编码进模型代码。数据预处理部分应负责产生这些模型入口可读取的数据文件和 metadata。

## 需要合并 agent 注意的点

- 目录名 `PCR-Net` 和 `U-Net` 中包含连字符，适合直接脚本执行，不适合 Python dotted import。若完整项目需要统一 package import，可以新增无连字符 wrapper，但不要破坏现有入口。
- `base_line` 拼写沿用当前迁移结构；合并时如要改名，需要同步更新入口路径和文档。
- `ResAttUNet` 默认结构不包含 CBAM；不要从旧工程恢复 CBAM。
- 训练入口可使用 torchvision 预训练 ResNet34。若目标环境无法联网或无缓存，应传入 `--no-pretrained-backbone`。
- 测试和可视化入口默认不需要预训练 backbone 初始化，因为会加载 checkpoint；对应参数默认关闭。
- 当前迁移代码做过静态语法检查和入口参数检查，但未在真实数据上运行训练、测试或出图。合并后需要在数据路径稳定后做 smoke test。
- 不要把旧工程中的生成图表、模型权重、catboost_info、历史 parquet 或实验日志作为源码迁入。
- 注释应保持简洁英文，只在关键逻辑处解释；不要恢复研究期旧注释。

## 推荐合并顺序

1. 先合并数据预处理项目的输出契约：metadata、static data、truth label、base temperature、guidance map、tabular parquet。
2. 再接入表格模型的数据构建、训练和空间推理入口。
3. 接入神经网络 dataset，确认四种泛化数据集能正常索引。
4. 接入 PCR-Net、baseline U-Net 和消融训练入口。
5. 最后接入统一测试和样本级可视化，用小 batch 或少量样本做 smoke test。
