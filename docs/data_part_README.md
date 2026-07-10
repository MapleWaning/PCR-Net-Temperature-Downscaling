# 数据处理与物理基底根目录

本目录承载迁移后的数据处理流水线和物理基底生成逻辑，后续会与模型训练项目合并成完整开源项目。这里的代码只负责把站点级原始/半手动数据加工成训练侧可直接消费的标准张量，不包含模型训练、评估或推理代码。

## 总体结构

```text
configs/
  default.yaml
  example_case.yaml
  physical.yaml
scripts/
  run_pipeline.py
  run_physical.py
src/
  data_pipeline/
    cli.py
    config.py
    pipeline.py
    registry.py
    common/
    modules/
    outputs/
  physical_base/
    cli.py
    config.py
    pipeline.py
    factors.py
    tbase.py
    validation.py
    physics/
docs/
  manual_downloads.md
```

`data_pipeline` 是 DATA 方案迁移后的统一数据流水线，负责 GSOD、Albedo、ERA5、LST、DEM、CCI 六类数据的计划、处理、输出和报告。

`physical_base` 是 PHYSICAL 方案迁移后的物理基底模块，读取 DATA 流水线产出的 metadata、DEM elevation 和 ERA5 temperature，再生成物理先验矩阵与逐年 advanced temperature base。

`configs` 中的路径均为示例路径。合并到训练项目时，应保留配置驱动方式，不要把本机绝对路径重新写回代码。

## 入口命令

数据流水线：

```bash
python -m data_pipeline.cli plan --config configs/example_case.yaml
python -m data_pipeline.cli run --config configs/example_case.yaml
```

脚本入口等价：

```bash
python scripts/run_pipeline.py plan --config configs/example_case.yaml
python scripts/run_pipeline.py run --config configs/example_case.yaml
```

物理基底：

```bash
python -m physical_base.cli plan --config configs/physical.yaml
python -m physical_base.cli run-factors --config configs/physical.yaml
python -m physical_base.cli run-tbase --config configs/physical.yaml
python -m physical_base.cli run --config configs/physical.yaml
```

## 核心数据契约

metadata CSV 是所有站点顺序的唯一来源。后续训练项目合并时，任何 dataset、dataloader 或特征拼接逻辑都应以 metadata 行顺序作为站点维度顺序。

关键约束：

- 站点数只能由 metadata 行数得到，禁止硬编码 `60`、`118` 或其他固定值。
- 所有模块的第一个站点维度必须严格对应 metadata CSV。
- 所有输出路径、原始数据路径、临时目录和年份范围必须来自配置。
- `plan` 用于检查输入和生成手动下载清单；`run` 才执行处理。
- 大型真实下载和完整数据处理不应作为普通单元测试的一部分。

标准输出形状：

```text
metadata/high-quality-meta.csv
gsod/weather_data.npy                    (n, days, 3)
albedo/Albedo_100m_YYYY_MM.npy           (n, 128, 128)
era5/era5_t2m_YYYY.h5                    (days, n, 4, 128, 128)
lst/lst_YYYY.h5                          (days, n, 3, 128, 128)
dem/dem_elevation.npy                    (n, 128, 128)
dem/dem_slope.npy                        (n, 128, 128)
dem/dem_aspect_sin.npy                   (n, 128, 128)
dem/dem_aspect_cos.npy                   (n, 128, 128)
cci/GLC_FCS30D_YYYY.npy                  (n, 128, 128)
physical/factors/*.h5                    see physical manifest
physical/t_base/t_base_advanced_YYYY.h5  (days, n, 4, 128, 128)
reports/manifest.json
reports/validation_report.csv
reports/physical_manifest.json
reports/physical_validation_report.csv
```

## DATA 模块说明

`gsod` 负责筛选 NOAA/ISD 站点、下载 GSOD 日数据、统计完整性、生成标准 metadata，并输出 `weather_data.npy`。温度列为 `TEMP, MAX, MIN`，单位转换为 Celsius，短缺口线性插值，长缺口保留 NaN。

`albedo` 使用 Google Earth Engine 的 Landsat Collection 2 Level 2 数据，保留 `QA_PIXEL` 掩膜、短波 albedo 公式、当月 median、前后月 fallback、全年 fallback 和 `-9999` 填充值。

`era5` 使用 CDS API 下载 ERA5 single levels `2m_temperature`，每日四个时次 `00/06/12/18`，插值到每站 `128x128 @ 100m` AEQD 网格，输出摄氏度 HDF5。

`lst` 是半手动模块。`plan` 生成站点-年份 GeoTIFF 清单，用户准备本地 MODIS LST GeoTIFF 后，`run` 将其 reshape、缺失修补、双线性上采样并写入 HDF5。

`dem` 根据 metadata 计算 SRTM CGIAR 5x5 度图幅，支持下载、解压、拼接，再按站点 AEQD 裁切，输出 elevation、slope、aspect sin/cos。

`cci` 是手动输入模块。`plan` 生成 CASEarth/GLC_FCS30D 分幅文件名清单；`run` 使用 `WarpedVRT` 以众数重采样到 EPSG:3857 网格，按 LUT 映射到 10 类，做时间前向填充和最近邻空间补洞。

## Physical 模块说明

`physical_base` 不重新处理 GSOD、DEM 或 ERA5 主数据。它只读取 DATA 阶段的标准结果：

```text
metadata/high-quality-meta.csv
dem/dem_elevation.npy
era5/era5_t2m_YYYY.h5
```

此外还需要物理模块自己的 ERA5 辅助输入：

```text
physical_raw.era5_surface_geopotential
physical_raw.era5_pressure_monthly
```

`run-factors` 输出：

```text
H_DEM_100m.h5
H_ERA5_avg.h5
Slope_100m.h5
Aspect_100m.h5
Basin_Mask_100m.h5
Gamma_Monthly_100m.h5
```

`run-tbase` 按年读取 `era5_t2m_YYYY.h5`，叠加月尺度高程递减、冬季盆地逆温和坡向辐射强迫，输出 `t_base_advanced_YYYY.h5`。输出 shape 与对应 ERA5 输入一致。

## 合并时需要注意

训练项目合并时，建议把本目录作为数据处理与物理基底子包保留，不要把模块重新拆回一次性脚本。

需要重点保护的边界：

- `data_pipeline.common.raster.station_aeqd_grids` 同时服务 ERA5 和 physical，请勿让两边出现不同网格生成逻辑。
- metadata 规范化集中在 `data_pipeline.common.station`，训练侧如需站点 ID，也应复用或保持兼容。
- CCI LUT 是明确数据契约，类别编号会影响训练标签/特征含义。
- ERA5 输出已经是 Celsius，physical 不再做 Kelvin 转换。
- `climatology_years` 是配置项，不能在代码中固定训练年份。
- 手动输入路径应继续通过 `manual_inputs` 和 `physical_raw` 配置传入。

## 验证策略

当前迁移代码提供轻量的 shape 和 manifest 报告：

```text
reports/manifest.json
reports/validation_report.csv
reports/physical_manifest.json
reports/physical_validation_report.csv
```

后续训练项目接入时，优先增加小样本 fixture 或 mock 文件测试，用来验证：

- metadata 行数与所有站点维度一致。
- 年份天数为 `365/366`。
- HDF5 dataset 名固定为 `data`。
- 物理 t_base 输出 shape 与 ERA5 输入完全一致。
- CCI 类别值范围为 `0..10`，清洗后预期应尽量无 `0`。
