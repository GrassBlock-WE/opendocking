[English](README.md) · **中文**

```text
 ██████╗ ██████╗ ███████╗███╗   ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║
╚██████╔╝██║     ███████╗██║ ╚████║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝
    ██████╗  ██████╗  ██████╗██╗  ██╗██╗███╗   ██╗ ██████╗
    ██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝██║████╗  ██║██╔════╝
    ██║  ██║██║   ██║██║     █████╔╝ ██║██╔██╗ ██║██║  ███╗
    ██║  ██║██║   ██║██║     ██╔═██╗ ██║██║╚██╗██║██║   ██║
    ██████╔╝╚██████╔╝╚██████╗██║  ██╗██║██║ ╚████║╚██████╔╝
    ╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
```

[English](README.md) · **简体中文**

**一套完整的开源分子对接与虚拟筛选工具链 —— 从原始 PDB 结构直达结合模式分析结果。**

* **纯 Rust 内核** —— Vina、Vinardo 与 AutoDock 4 三套力场，全部提供**解析梯度**；三线性插值亲和力网格；蒙特卡洛迭代局部搜索；岛屿模型拉马克遗传算法；Solis-Wets 自适应局部搜索。用 `rayon` 并行，并可通过 `wgpu` + WGSL 在 GPU 上加速。
* **Python 原生接口** —— PyO3 绑定，与 NumPy 零拷贝互通；支持协作式暂停 / 恢复 / 中止，以及多配体并行批量对接。
* **端到端流程** —— 受体清洗与 pH 自适应质子化；配体构象生成、力场能量最小化、点击锁定可旋转键；无配体盲测口袋；交互式搜索盒；对称感知 RMSD 聚类；六类相互作用（距离阈值可自选）；二维相互作用拓扑图；Excel / CSV 报表。
* **三维工作台** —— 卡通、带状、空间填充、球棍、键线式等多种显示样式；可点击的氨基酸序列标尺；选区原子面板；作用位点高亮并精确指出成对原子。中英双语界面。
* **验证结果** —— 605 个 Python 测试与 116 个 Rust 测试全部通过；苯甲脒重新对接进胰蛋白酶（PDB 3PTB），不做叠合的 top-pose RMSD 为 **1.13 Å**。

许可证：GPL-3.0-or-later。

---

## 功能

**受体预处理**（`python/odock/chem/receptor.py`、`charges.py`、`flex.py`）

* 异质原子（HETATM）分类与清洗：按残基名模式剔除溶剂；按与配体的距离保留结构水；识别并剥离共结晶配体（同时可作为自对接验证的参考构象）；按清单区分金属阳离子、游离反离子与辅酶因子，可逐项勾选保留。
* 缺失原子检测：比对 20 种标准氨基酸、MSE/SEC 与 10 种核苷酸的模板，报告不完整的残基。
* pH 自适应质子化（默认 7.4，0–14）：组氨酸按局部环境判定 HID / HIE / HIP；Asp/Glu 为 −1，Lys/Arg 为 +1；只增删 N/O/S 上的氢。
* 非极性氢合并且把其电荷累加回母体碳，保证总电荷守恒。
* Gasteiger 电荷与 AD4 原子类型映射（含芳香碳 `A`、`SA`、卤素，以及无类型氢的 `W` 哨兵）。
* 柔性残基导出为含 `BEGIN_RES` / `BRANCH` / `END_RES` 的柔性受体 PDBQT。

**配体处理**（`python/odock/chem/ligand.py`、`filters.py`）

* SMILES / SDF / MOL2 / MOL / PDB / PDBQT 单分子与批量导入，二维结构用 ETKDGv3 生成三维构象。
* MMFF94 / MMFF94s / UFF 能量最小化（200–1000 步），力场无法参数化时自动回退并记录原因。
* 旋转键感知，逐条排除酰胺键、炔键、芳香/共轭环内键、末端甲基与叔丁基等无意义旋转，并可解释触发的是哪条规则。
* 成药性过滤：Lipinski 五规则、Veber 规则、PAINS（RDKit FilterCatalog 480 条，另有 45 条自备 SMARTS 回退）。

**口袋与网格**（`python/odock/pocket.py`）

* 以共结晶配体包络、以指定残基质心对齐，或无配体时**盲测**表面空腔并按体积与埋藏度排序，一键把搜索盒对准选定口袋。
* 交互式搜索盒：中心 / 尺寸（0.001 Å 精度）/ 步长（0.1–1.0 Å 无级调节）；盒子可半透明显示，不透明度可调。
* 三线性插值势能网格，含解析梯度与出界惩罚。

**对接引擎**（`crates/dock-core`、`crates/dock-py`）

* 三套力场：Vina（高斯吸引 ×2、二次斥力、线性疏水、定向氢键）、Vinardo、AutoDock 4（LJ 12-6/12-10、距离介电库仑、Myers-Free 去溶剂化）。
* 搜索器：蒙特卡洛 ILS、岛屿模型 LGA、以及用 Solis-Wets 替代 BFGS 的 LGA。
* 参数可控：穷举度、构象数、能量窗口、随机种子、岛屿数、种群规模、代数、线程数。
* 运行中可暂停 / 恢复 / 中止（中止后迅速返回已找到的最优构象），也可批量并行处理多个体系。

**分析**（`python/odock/analysis.py`、`report.py`）

* 对称感知 RMSD 聚类（自实现匈牙利算法，无额外依赖），按 2.0 Å 截断给出空间优势簇与代表构象。
* 六类相互作用：氢键、盐桥、π–π 堆积（区分面对面 / T 型）、阳离子–π、疏水接触、立体碰撞；**每类的距离阈值都可在界面中自行设置**。
* 二维相互作用拓扑图（SVG）、构象回放动画、Excel / CSV 报表。

**三维工作台**（`python/odock/gui`）

* 蛋白质 8 种样式：卡通（α-螺旋为盘绕带、β-折叠为箭头）、带状、管状、球体、**空间填充（1:1 范德华半径）**、棍状、球棍、点阵；配体 4 种：球棍、棍状、**键线式**、球体。
* 化学键按 PyMOL 式价态模型识别（共价半径 + 价态上限 + 短键优先），金属与惰性气体永不连键，受体只允许肽键与二硫键跨残基；可与 RDKit 的连通性算法逐键比对。
* 氨基酸序列标尺：单字母简称、每 5 个残基一个刻度并标残基编号、配体与离子以原名列于末尾；点击即选中对应残基。
* 运行对接后点击「显示相互作用」，作用残基与成对原子以球棍高亮、其余结构压暗，接触虚线加粗显示。
* 换构象时相机与受体完全不动，只有配体移动，便于逐个比对。

## 环境要求

| 组件 | 版本 / 说明 |
|---|---|
| Rust 工具链 | 1.87 或更新（`cargo`） |
| Python | 3.9 或更新 |
| 构建 | `maturin`（必需，用于编译 Rust 扩展） |
| 必需依赖 | `rdkit`、`numpy` |
| 可选依赖 | `PyQt6` + `moderngl`（三维工作台）、`openpyxl`（XLSX 报表） |

> **重要**：`odock` 的 Python 包依赖编译出的 `_odock` 扩展，源码仓库中不包含编译产物。
> 克隆之后必须先执行 `maturin develop --release`，否则 `import odock` 会失败。

## 快速开始

```bash
# 1. 建立虚拟环境并安装 Python 侧依赖
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install maturin rdkit numpy

# 2. 把 Rust 扩展编译进该环境（必需）
maturin develop --release

# 3. 可选：三维工作台与 Excel 报表
pip install PyQt6 moderngl openpyxl

# 4. 打开三维工作台（裸 odock 命令只打印命令行帮助）
odock gui
```

最小可跑通的完整流程：

```bash
odock prepare receptor receptor.pdb receptor.pdbqt --strip BEN
odock prepare ligand   ligand.sdf   ligand.pdbqt
odock box   --ligand ligand.pdbqt --buffer 8 --out box.json
odock dock  -r receptor.pdbqt -l ligand.pdbqt --box box.json \
            -e 16 --seed 42 -o poses.pdbqt
```

## 命令行参考

| 命令 | 作用 |
|---|---|
| `odock` | 打印命令行帮助 |
| `odock info` | 显示内核、力场与 GPU 信息 |
| `odock prepare receptor` | 准备刚性受体 PDBQT（可清洗、加氢、剥离指定残基） |
| `odock prepare ligand` | 准备配体 PDBQT（可来自 SMILES / SDF / MOL2 / PDB） |
| `odock box` | 由配体、残基或坐标构造搜索盒 |
| `odock dock` | 执行对接搜索 |
| `odock score` | 对给定位置打分，不做搜索 |
| `odock pocket` | 无配体盲测口袋并排序输出 |
| `odock filter` | Lipinski / Veber / PAINS 成药性报告 |
| `odock cluster` | 对称感知的构象 RMSD 聚类 |
| `odock interactions` | 受体–配体相互作用剖析 |
| `odock diagram` | 生成二维相互作用拓扑图（SVG） |
| `odock report` | 导出结果报表（XLSX / CSV） |
| `odock fetch` | 从 RCSB 下载结构 |
| `odock export` | 导出 GPF / DPF / Vina 配置 / 清洗后 PDB |
| `odock split` | 把多模型 PDBQT 拆分为单模型文件 |
| `odock gui` | 打开三维工作台 |

## Python API

```python
import odock

# —— 预处理 ——
receptor_mol, receptor_pdbqt, _ = odock.prepare_receptor("receptor.pdb")
ligand_mol, ligand_pdbqt, _     = odock.prepare_ligand("ligand.sdf")

# —— 搜索盒（或使用 pocket.find_pockets 做盲测）——
box = odock.box_from_ligand(ligand_mol, buffer=8.0)

# —— 对接 ——
result = odock.dock(receptor_pdbqt, ligand_pdbqt, box,
                    scoring="vina", exhaustiveness=16, seed=42,
                    search="lga_solis")
print(result.table())

# —— 分析 ——
from odock import analysis, pocket, report, filters

contacts = analysis.profile_interactions(receptor_atoms, ligand_atoms)
print(analysis.interaction_summary(contacts, receptor_atoms, ligand_atoms))
print(pocket.find_pockets(receptor_atoms)[0])
print(filters.drug_like(ligand_mol))
report.write_xlsx("results.xlsx", result)

analysis.interaction_diagram_svg(receptor_atoms, ligand_atoms, contacts,
                                 path="interactions.svg")
```

## 验证

```bash
cargo test --workspace                  # 116 个内核测试
python -m pytest tests -q               # 605 个 Python 测试
python tests/validate_3ptb.py           # 晶体结构复现验收
```

| 检查项 | 结果 |
|---|---|
| Rust 内核测试 | **116 passed**（另含 GPU 后端测试） |
| Python 测试 | **605 passed, 1 skipped** |
| 三维工作台模拟操作 | **48 / 48 步通过**（真实 Qt 鼠标键盘事件驱动） |

**晶体结构复现** —— 从原始 PDB 出发自建受体与配体、随机起始构象，重新对接 PDB 3PTB（牛胰蛋白酶 + 苯甲脒）：

```text
mode |   affinity |  RMSD (no fit) |  RMSD (fitted)
-----+------------+----------------+---------------
   1 |     -6.213 |          1.133 |          1.133
   2 |     -6.191 |          1.916 |          1.916
   …
best-pose RMSD to the crystal structure: 1.133 Å
RESULT: PASS — the top pose reproduces the experimental binding mode
```

条件：穷举度 16、随机种子 42、网格 150 920 点 / 3 MB、耗时约 3 秒。同一受体上的盲测口袋探测把真实的 S1 位点排在第 2 位（体积 192 Å³，距晶体配体质心 4.6 Å，由 GLN192 / SER195 / SER214 / TRP215 / GLY216 围成）。

## 仓库结构

```text
Cargo.toml                   Rust 工作区
pyproject.toml               maturin 打包配置
crates/dock-core/            内核（纯 Rust，#![deny(unsafe_code)]）
    src/math.rs              四元数与刚体变换
    src/rng.rs               确定性 PCG-XSH-RR 随机数
    src/atom.rs              AD4 与 X-Score 原子类型及参数表
    src/molecule.rs          原子、键与基于图的键感知
    src/kinematics.rs        刚性簇、扭转树、正向运动学
    src/cancel.rs            协作式暂停 / 中止令牌
    src/scoring/             势能项、梯度、亲和力网格
    src/search/              BFGS、蒙特卡洛 ILS、岛屿 GA、Solis-Wets
    src/io/pdbqt.rs          PDBQT 读写
    src/gpu/                 可选 wgpu 后端（grid.wgsl）
    src/docking.rs           编排层
crates/dock-py/              PyO3 + NumPy 绑定
python/odock/                Python 包
    chem/                    受体与配体化学
    analysis.py              RMSD 聚类与相互作用剖析
    pocket.py                盲测口袋
    filters.py               成药性过滤
    export.py / fetch.py     格式导出与结构下载
    report.py                报表
    cli.py                   命令行界面
    gui/                     PyQt6 + ModernGL 三维工作台
tests/                       Python 测试与验收脚本
demo/                        端到端演示数据（3PTB、1M17）
docs/                        架构、数据结构、打分推导、使用指南、验证
examples/make_demo.py        重新生成 demo/ 下的全部数据
```

## 文档

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) —— 目录规划、依赖关系、模块职责、数据流与线程模型
* [`docs/DATA_STRUCTURES.md`](docs/DATA_STRUCTURES.md) —— 全部核心数据结构及其不变量
* [`docs/SCORING.md`](docs/SCORING.md) —— 打分公式逐项推导与解析梯度
* [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) —— 完整使用指南
* [`docs/VALIDATION.md`](docs/VALIDATION.md) —— 验证结果与**未经验证的部分**

## 参与贡献

见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。版本变更见 [`CHANGELOG.md`](CHANGELOG.md)。

## 许可证

Copyright (C) 2024-2026 The OpenDocking Project.

本项目以 **GNU 通用公共许可证第 3 版或更新版本**（GPL-3.0-or-later）发布，全文见 [`LICENSE`](LICENSE)。

### 上游署名

OpenDocking 是独立实现，算法与文件格式参考了下列以自由许可发布的项目，并在相应源文件头部保留署名：

* **AutoDock Vina** —— Copyright (c) 2006-2010, The Scripps Research Institute，Apache-2.0。
* **AutoDock 4** —— GPL，AD4 力场、LGA 与 Solis-Wets 局部搜索、GPF/DPF 格式的参考。
* **AutoDock-GPU** —— LGPL-2.1，异构加速架构的参考。
* **Meeko** —— LGPL-2.1，配体预处理行为的参考。

**未使用任何 AutoDockTools（ADT / MGLTools）专有代码**：文件解析、扭转树构建与全部前后处理均为基于现代开源规范（RDKit 与自有实现）从零编写。
