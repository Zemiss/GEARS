# 用 VCC 训练和评估 GEARS

新增入口为 `scripts/vcc.py`，调用原版 `PertData` 和 `GEARS.train()`，网络、损失函数、优化器和验证选模逻辑保持原样。以下命令在服务器执行，本地无需训练。

## 环境

按 GEARS 官方方式安装与服务器 PyTorch/CUDA 匹配的 PyG，再安装本地仓库：

```bash
cd /data/xiebo/mycode/GEARS
python -m pip install -e .
```

建议 Python 3.11，依赖版本以仓库 `requirements.txt` 为准。`Lingshu-Cell` 只作为基因词表、Ensembl 映射的只读来源，不需要安装或修改它。

## 1. 准备数据

```bash
python scripts/vcc.py prepare \
  --data-root /data/dataset/lingshucell/vcc \
  --work-dir /data/dataset/gears/vcc
```

默认读取：

- `train/adata_Training.h5ad`、`validation/adata_Validation.h5ad` 的表达矩阵。
- `test/pert_counts_Test.csv` 的预测目标及细胞数量；不读取 `test/adata_Test.h5ad`。
- 相邻 `Lingshu-Cell/data/vocabs/legacy/tokenizer_vcc_cond_dense_gene1tok.json` 的全部 18,080 个基因及顺序。
- 相邻 `Lingshu-Cell/data/vcc/gene_id_name.csv` 的 Ensembl → symbol 映射。

约定和可选参数：

- 默认 `X` 是原始 counts，按选定基因面板逐细胞 `normalize_total(10000) + log1p`。counts 校验不通过会报错；已归一化表达必须显式指定 `--input-scale lognorm`，此时不再次变换。counts 在 layer 时使用 `--layer counts`。
- 默认扰动列为 `obs.target_gene`，单一背景为 H1；可用 `--target-key`、`--cell-type` 指定。基因可在 `var.gene_name`、`gene_symbols`、`symbol`、`feature_name` 或 `var_names`，Ensembl 版本后缀可映射。
- 对照转换为 `ctrl`，单扰动转换为 `GENE+ctrl`。训练集必须包含对照；验证集对照不进入训练或预测的基线池。
- 保留官方训练/验证目标划分；train/val/test 扰动重叠时直接报错。不会调用随机 `simulation` 划分。验证集真实响应仅用于验证指标和 checkpoint 选择。
- 默认完整保留 Lingshu 面板，不筛高变基因。`--gene-panel` 可改为单列基因列表（CSV/txt，可带 `gene_name` 表头），或 Lingshu 词表 JSON。缺失基因、重复 symbol、缺 GO 注释均报错，不补零或静默删目标。
- 通过 GEARS 原生 `gene_set_path` 使用 train/val/test 的目标身份构建自定义 GO 扰动图；测试目标身份是元数据，不使用测试表达。相比默认大图，图节点限制为 VCC 目标基因，算法不变。首次准备下载 GEARS 的 GO 注释；训练时缓存自定义图和训练数据的共表达图。
- 可用 `--train-h5ad`、`--val-h5ad`、`--conditions`、`--gene-map` 覆盖路径。工作目录必须为空；重做预处理请使用新目录，以免旧的细胞图和网络缓存混入。

准备结果包含 `perturb_processed.h5ad`、`data_pyg/cell_graphs.pkl`、`split.pkl`、训练对照 `control.h5ad`、`conditions.csv`、`panel.txt` 和 `vcc.json`。最后一个文件是完成标记及数据配置记录。

## 2. 训练

```bash
python scripts/vcc.py train \
  --work-dir /data/dataset/gears/vcc \
  --model-dir /data/xiebo/mycode/models/gears/vcc \
  --device cuda:0 --batch-size 16 --epochs 20 --seed 1
```

原版训练根据验证集 DE20 MSE 选择最优 checkpoint；保存 `config.pkl`、`model.pt` 和配套 `vcc.json`。不会执行原版随机测试集评估。`--hidden-size` 默认 64，`--lr` 默认 0.001。训练目录必须为空。

原版 GEARS 会缓存每个细胞的稠密输入/输出图，并在训练时评估训练集；全量 18,080 基因需要较多 CPU 内存、磁盘和显存。仅两个 float32 表达向量的存储下限为 `细胞数 × 基因数 × 8` 字节，另有图对象、共表达计算、评估数组及模型开销。本次没有重写这些原版流程；降低 batch size 只降低部分显存开销，不会消除 CPU 缓存开销。

## 3. 导出逐细胞预测

```bash
python scripts/vcc.py predict \
  --work-dir /data/dataset/gears/vcc \
  --model-dir /data/xiebo/mycode/models/gears/vcc \
  --conditions /data/dataset/lingshucell/vcc/evaluation/conditions.csv \
  --output /data/xiebo/mycode/outputs/gears/prediction.h5ad \
  --device cuda:0 --batch-size 16 --seed 1
```

`--conditions` 使用与 Lingshu-Cell 相同的条件表，包含 `target_gene,n_cells`，可带 `cell_type`；省略则使用准备时保存的官方 test 条件表。条件表中的目标必须已在准备时的 GO 图中。每个目标按 `n_cells` 从训练对照中有放回抽样，并逐批预测，不对细胞取均值。

输出 H5AD 只有预测扰动细胞，基因顺序与训练一致，`obs.target_gene` 恢复为 `GENE`，`obs.cell_type=H1`。`X` 保留连续 lognorm 表达，不伪装成 counts；负预测截为 0 并记录数量，NaN/Inf 直接报错。文件内声明 `uns['evaluation']['expression_scale']='lognorm'`，兼容现有 `evaluate`。预测通过磁盘临时矩阵汇总；需要为临时矩阵和最终 H5AD 留出空间。现有预测文件不会被覆盖。

## 4. 使用共用参考进行评估

```bash
cd /data/xiebo/mycode/evaluate
python -m evaluate \
  --pred-path /data/xiebo/mycode/outputs/gears/prediction.h5ad \
  --truth-path /data/dataset/lingshucell/vcc/evaluation/truth.h5ad \
  --control-path /data/dataset/lingshucell/vcc/evaluation/control.h5ad \
  --output-dir /data/xiebo/mycode/results/gears
```

复用已有 truth/control/panel/conditions，不为 GEARS 重建评估参考。预测必须覆盖共用 truth 的所有基因；如果共用参考含 Lingshu 面板外基因，需要先统一实验面板。GEARS 与 Lingshu-Cell 的模型和预训练经历不同，这里对齐的是 VCC 数据、预测目标、基因面板与评估协议。

## 轻量检查

```bash
python -m pip install pytest
python -m pytest -q tests/test_vcc.py
```

测试使用小型 AnnData 和模拟模型，覆盖基因重排、Ensembl 映射、counts layer、尺度校验、验证对照隔离、划分泄漏拒绝、缺失基因/GO、无测试表达的数据准备，以及分批逐细胞 H5AD 导出；不执行模型训练。
