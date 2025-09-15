## **A First-Principles Derivation of Neural Scaling Laws**

---

## **English Version**

This repository contains the official code for the paper **"A First-Principles Derivation of Neural Scaling Laws: A Unified Theory of Hebbian Dynamics and Experimental Contexts"**. We demonstrate how macroscopic scaling laws (D-Scaling, N-Scaling, Chinchilla's Law) emerge from a single microscopic Hebbian dynamical equation, driven by the antagonistic dynamics between generalization and memorization.

#### **Theoretical Framework**

The core of our theory is a single Hebbian-inspired Ordinary Differential Equation (ODE) that models the evolution of a single weight $W\_i$. This equation describes a fundamental competition between a **Learning Term** (creation) and a **Forgetting Term** (destruction):

$$\frac{dW_i}{dt} = \underbrace{\eta \cdot \text{corr}(f_i, C) \cdot W_i}_{\text{Learning Term}} - \underbrace{\lambda W_i}_{\text{Forgetting Term}}$$

We argue that all seemingly contradictory scaling laws are emergent macroscopic patterns of this single microscopic dynamic, depending on the experimental context. We introduce two key metrics to measure the system's internal state:

  * **Weighted Topological Semantic Entropy ($H'\_{tse}$)**: Measures the system's **abstraction efficiency** or generalization cost.
  * **Weighted Semantic Information Entropy ($H'\_{sie}$)**: Measures the system's **redundancy and robustness**.

The key insight is that the influence of **dataset size (D)** and **model capacity (N)** are fundamentally asymmetric:

  * **D (External Input)** drives the system's **temporal evolution**. We analyze its effect by studying the solution form of the ODE over time.
  * **N (Internal Structure)** defines the system's **solution space geometry**. We analyze its effect by studying the statistical properties of this space.

-----

#### **Key Findings**

The same Hebbian engine produces two diametrically opposed, yet predictable, macroscopic patterns, which perfectly match experimental results.

##### **1. D-Scaling (Data Complexity ≫ Model Capacity)**

In this resource-limited context, the **Forgetting Term** dominates, forcing the system to generalize and compress information.

  * $H'\_{tse}$ (abstraction cost) **rises** following a **power law**.
  * $H'\_{sie}$ (robustness) **decays** following an **exponential law**.
<img width="7168" height="4462" alt="vit_D" src="https://github.com/user-attachments/assets/2803eaf4-6349-495c-bfff-ebc25e4835d5" />
(result on Vit)

##### **2. N-Scaling (Model Capacity ≫ Data Complexity)**

In this resource-abundant context, the **Learning Term** dominates, forcing the system to robustly memorize the fixed task.

  * $H'\_{tse}$ (abstraction cost) **decays** following a **power law**.
  * $H'\_{sie}$ (robustness) **grows** following a **logarithmic law**.
<img width="7168" height="4462" alt="vit_N" src="https://github.com/user-attachments/assets/f0cbfb19-3228-4271-a136-5612d32dfe85" />
(result on Vit)

-----


#### **Repository Structure**

```
.
├── assets/                         # Images for README
├── D-Scaling/                      # D-Scaling logic modules (MLP, CNN, ViT)
│   ├── MLP_D_logic.py
│   ├── CNN_D_logic.py
│   └── VIT_D_logic_EN.py
├── N-Scaling/                      # N-Scaling logic modules (MLP, CNN, ViT)
│   ├── MLP_N_logic.py
│   ├── CNN_N_logic.py
│   └── VIT_N_logic.py
├── D_Scaling_EN.ipynb              # Main notebook for D-Scaling experiments
├── N_Scaling_EN.ipynb              # Main notebook for N-Scaling experiments
├── publication_figure_generator.py # Script to generate final figures from CSV
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

#### **How to Run the Experiments**

1.  **Setup Environment**
    Clone the repository and set up the Python environment. We recommend using Conda.

    ```bash
    git clone https://github.com/alicethegod/First-Principles-Scaling-Laws.git
    cd First-Principles-Scaling-Laws
    conda create -n scaling_laws python=3.9
    conda activate scaling_laws
    pip install -r requirements.txt
    ```

    *(Note: You will need to generate the `requirements.txt` file via `pip freeze > requirements.txt`)*

**Quick Demo: logarithmic_dynamics_demo.ipynb (3 mins)**

To quickly verify the core dynamical prediction of our theory, you can run the minimal experiment. This demo trains a small MLP on a toy dataset and reveals that the internal entropy metrics ($H'\_{tse}$ and $H'\_{sie}$) evolve following precise logarithmic laws during the compression phase of learning.

2.  **Run the Main Experiments**
    Open the Jupyter notebooks `D_Scaling_EN.ipynb` or `N_Scaling_EN.ipynb`. Select the desired logic module (e.g., `VIT_D_logic_EN.py` for D-Scaling with a ViT) within the notebook and run all cells. This will execute a full experimental sweep and save the results as a `.csv` file.

3.  **Generate Publication-Quality Figures**
    Use the `publication_figure_generator_EN.py` script to generate the final figures from the `.csv` file produced in the previous step.

    ```bash
    python publication_figure_generator.py path/to/your_results.csv
    ```

    The script will automatically generate a high-resolution summary figure in the same directory as the input CSV.

-----

#### **How to Cite**

If you find this work useful in your research, please consider citing our paper:

```
@misc{Liu2025ScalingLaws,
  author       = {Liu, Zhangchi},
  title        = {A First-Principles Derivation of Neural Scaling Laws: A Unified Theory of Hebbian Dynamics and Experimental Contexts},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17118793},
  url          = {https://doi.org/10.5281/zenodo.17118793}
}
```

---

## **中文版本**

本仓库为论文 **《神经缩放定律的第一性原理推导：一个关于赫布动力学与实验情境的统一理论》** 的官方代码。我们展示了宏观的缩放定律（D-Scaling, N-Scaling, Chinchilla定律）是如何从一个单一的微观赫布动力学方程，在泛化与记忆的拮抗动力学驱动下涌现出来的。

#### **理论框架**

我们理论的核心，是一个受赫布学习启发的常微分方程（ODE），它描述了单个权重 $W\_i$ 的演化。这个方程刻画了**学习项**（创造）与**遗忘项**（毁灭）之间最根本的竞争：

$$\frac{dW_i}{dt} = \underbrace{\eta \cdot \text{corr}(f_i, C) \cdot W_i}_{\text{学习项}} - \underbrace{\lambda W_i}_{\text{遗忘项}}$$

我们主张，所有看似矛盾的缩放定律，都是这一微观动力学在不同实验情境下的宏观涌现模式。我们引入两个核心度量来衡量系统的内部状态：

  * **加权拓扑语义熵 ($H'\_{tse}$)**: 衡量系统的**抽象效率**或泛化成本。
  * **加权语义信息熵 ($H'\_{sie}$)**: 衡量系统的**冗余度和鲁棒性**。

本理论的关键洞见在于，**数据量 (D)** 和**模型容量 (N)** 的影响是根本不对称的：

  * **D (外部输入)** 驱动系统的**时间演化**。我们通过研究ODE解随时间的变化来分析其效应。
  * **N (内部结构)** 定义系统的**解空间几何**。我们通过研究该空间随N变化的统计特性来分析其效应。

-----

#### **主要发现**

同一个赫布引擎，产生了两种截然相反但可被精确预测的宏观模式，这与实验结果完美吻合。

##### **1. D-Scaling (数据复杂度 ≫ 模型容量)**

在这个资源受限的情境下，**遗忘项**占据主导，迫使系统进行泛化和信息压缩。

  * $H'\_{tse}$ (抽象成本) 遵循**幂律上涨**。
  * $H'\_{sie}$ (鲁棒性) 遵循**指数下降**。
<img width="7168" height="4462" alt="vit_D" src="https://github.com/user-attachments/assets/d23266d7-8d3e-4154-bc4a-a26fec058819" />
（Vit 结果）

##### **2. N-Scaling (模型容量 ≫ 数据复杂度)**

在这个资源充裕的情境下，**学习项**占据主导，驱使系统对固定任务进行鲁棒的记忆。

  * $H'\_{tse}$ (抽象成本) 遵循**幂律下降**。
  * $H'\_{sie}$ (鲁棒性) 遵循**对数增长**。
<img width="7168" height="4462" alt="vit_N" src="https://github.com/user-attachments/assets/e0b9b378-14b4-4491-8246-b82d3b6dc9ad" />
（Vit 结果）

-----


#### **仓库结构**

```
.
├── assets/                          # README中使用的图片
├── D-Scaling/                       # D-Scaling 逻辑模块 (MLP, CNN, ViT)
│   ├── MLP_D_logic_EN.py
│   ├── CNN_D_logic.py
│   └── VIT_D_logic.py
├── N-Scaling/                       # N-Scaling 逻辑模块 (MLP, CNN, ViT)
│   ├── MLP_N_logic.py
│   ├── CNN_N_logic.py
│   └── VIT_N_logic.py
├── D_Scaling_EN.ipynb               # D-Scaling 实验的主 notebook
├── N_Scaling_EN.ipynb               # N-Scaling 实验的主 notebook
├── publication_figure_generator.py # 从CSV生成最终论文图表的脚本
├── requirements.txt                 # Python 依赖项
└── README.md                        # 本文件
```

#### **如何运行实验**

1.  **配置环境**
    克隆本仓库并配置Python环境，推荐使用 Conda。

    ```bash
    git clone https://github.com/alicethegod/First-Principles-Scaling-Laws.git
    cd First-Principles-Scaling-Laws
    conda create -n scaling_laws python=3.9
    conda activate scaling_laws
    pip install -r requirements.txt
    ```

    *(注意: 您需要先运行 `pip freeze > requirements.txt` 命令来生成您的依赖项文件)*
**快速入门演示: logarithmic_dynamics_demo.ipynb (3分钟)**

为了快速验证我们理论的核心动力学预测，您可以运行这个最小化的实验。该演示在一个玩具数据集上训练一个小型MLP，并揭示了其内部熵指标 ($H'\_{tse}$ 和 $H'\_{sie}$) 在学习的压缩阶段，精确地遵循着对数定律进行演化。

2.  **运行主实验**
    打开 Jupyter Notebook `D_Scaling_EN.ipynb` 或 `N_Scaling_EN.ipynb`。在 notebook 内部选择您希望使用的逻辑模块（例如，为ViT选择 `VIT_D_logic_EN.py`），然后运行所有代码单元。这将执行完整的实验扫描，并将结果保存为 `.csv` 文件。

3.  **生成论文级别的图表**
    使用 `publication_figure_generator.py` 脚本，处理上一步生成的 `.csv` 文件，以生成最终的论文图表。

    ```bash
    python publication_figure_generator.py path/to/your_results.csv
    ```

    脚本会自动在输入CSV文件所在的目录下，生成一张高分辨率的汇总图。

-----

#### **如何引用**

如果您在您的研究中发现本工作有用，请考虑引用我们的论文：

```
@misc{Liu2025ScalingLaws,
  author       = {刘章驰},
  title        = {神经缩放定律的第一性原理推导：一个关于赫布动力学与实验情境的统一理论},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17118793},
  url          = {https://doi.org/10.5281/zenodo.17118793}
}
```
