# MindSpeed-LLM 项目架构概览

## 1. 项目简介

MindSpeed-LLM（原名 ModelLink）是华为昇腾团队开发的**大语言模型分布式训练工具套件**，专为华为 Ascend NPU 硬件生态设计。项目提供从预训练、微调到推理、评估的端到端解决方案，支持 100+ 种 LLM 架构。

- **版本**：26.0.0dev0（商业版本 2.3.0）
- **许可证**：Apache 2.0
- **硬件要求**：Atlas 800T A2（Ascend NPU）

---

## 2. 顶层目录结构

```
MindSpeed-LLM/
├── pretrain_gpt.py          # GPT 预训练入口
├── pretrain_mamba.py         # Mamba 模型预训练入口
├── posttrain_gpt.py          # 后训练（微调）入口
├── rlhf_gpt.py               # RLHF 训练入口
├── train_fsdp2.py            # FSDP2 训练框架入口
├── inference.py              # 推理入口
├── evaluation.py             # 评估入口
├── preprocess_data.py        # 数据预处理
├── convert_ckpt.py           # 权重转换 v1
├── convert_ckpt_v2.py        # 权重转换 v2
│
├── mindspeed_llm/            # 核心 Python 包（410+ 文件）
├── configs/                  # 配置文件（YAML/JSON）
├── examples/                 # 各模型训练脚本示例（32+ 模型）
├── tests/                    # 测试（ut/st/poc/coverage）
├── docs/                     # 文档
├── ci/                       # CI/CD 脚本
├── requirements.txt          # Python 依赖
└── setup.py                  # 包构建配置
```

---

## 3. 核心包架构（mindspeed_llm/）

```
mindspeed_llm/
├── core/               # 底层：Transformer 基础构件
├── tasks/              # 中层：任务级实现
├── training/           # 高层：Megatron 训练框架
├── fsdp2/              # 高层：PyTorch FSDP2 训练框架
├── features_manager/   # 正交层：特性管理与注入
├── inference/          # 推理管道
├── legacy/             # 遗留兼容代码
└── mindspore/          # MindSpore 后端
```

### 分层关系图

```
┌──────────────────────────────────────────────────────┐
│              入口脚本（pretrain_gpt.py 等）            │
├──────────────────────────────────────────────────────┤
│  training/          │  fsdp2/                        │ ← 高层：训练框架
│  (Megatron 训练循环) │  (FSDP2 训练循环)               │
├──────────────────────────────────────────────────────┤
│  tasks/                                              │ ← 中层：任务实现
│  (模型定义/数据集/微调/评估/推理/检查点转换)             │
├──────────────────────────────────────────────────────┤
│  core/                                               │ ← 底层：基础构件
│  (Transformer/并行策略/优化器/数据集/嵌入)              │
├──────────────────────────┬───────────────────────────┤
│  features_manager/       │  正交切面：特性注册与注入     │
└──────────────────────────┴───────────────────────────┘
```

---

## 4. 核心模块详解

### 4.1 core/ — Transformer 基础构件

提供分布式训练的低级构建模块，与 Megatron-LM 深度集成。

| 子模块 | 职责 |
|--------|------|
| `transformer/` | Attention（Flash/ALiBi/MLA）、MLP、Transformer Block/Layer、MOE Router |
| `models/` | GPT 模型层规格、嵌入（Rotary/ALiBi）、RMSNorm、Loss 计算 |
| `tensor_parallel/` | 张量并行层实现，含 2D TP 变体 |
| `pipeline_parallel/` | 流水线并行调度（含 DualPipe） |
| `context_parallel/` | 上下文/序列并行 |
| `distributed/` | 参数与梯度缓冲区、梯度同步 |
| `optimizer/` | 分布式优化器、梯度裁剪 |
| `datasets/` | Megatron 数据集构建器（GPT Dataset、Indexed Dataset） |
| `high_availability/` | 容错、弹性训练、检查点 |

### 4.2 tasks/ — 任务级实现

基于 core/ 构建的面向具体任务的上层逻辑。

| 子模块 | 职责 |
|--------|------|
| `models/` | 100+ 模型的规格定义与适配 |
| `dataset/` | 数据集处理（Alpaca、ShareGPT、预训练格式等） |
| `posttrain/` | 微调实现：SFT、LoRA、QLoRA、LU-LoRA、DPO |
| `checkpoint/` | 检查点格式转换（HF ↔ Megatron） |
| `inference/` | 推理模块 |
| `evaluation/` | 评估基准（MMLU、C-Eval、BBH、HumanEval、GSM8K 等） |
| `preprocess/` | 数据预处理管道 |
| `megatron_adaptor_v2.py` | **关键桥梁**：将 features_manager 注册的特性适配到 Megatron |

### 4.3 training/ — Megatron 训练框架

基于 Megatron-LM 的主训练循环，是目前最成熟的训练后端。

| 文件 | 职责 |
|------|------|
| `training.py`（66KB） | 主训练循环、`pretrain()` 入口函数 |
| `arguments.py`（77KB） | 全量参数解析与验证 |
| `initialize.py` | Megatron 分布式环境初始化 |
| `checkpointing.py` | 检查点保存/加载 |
| `utils.py` | 训练工具函数 |

### 4.4 fsdp2/ — PyTorch FSDP2 训练框架

基于 PyTorch 原生 Fully Sharded Data Parallel v2 的新一代训练框架，作为 Megatron 的替代方案。

| 子模块 | 职责 |
|--------|------|
| `train/` | Trainer 层级：BaseTrainer → MindSpeedTrainer → PretrainTrainer / SFTTrainer |
| `models/` | 模型工厂、模型加载器、FSDP2 模型封装、模型注册表 |
| `data/` | 数据工厂、Tokenizer、模板处理 |
| `optim/` | 优化器与学习率调度器工厂 |
| `checkpoint/` | FSDP2 检查点管理 |
| `distributed/` | FSDP 分片、专家并行、上下文并行 |

### 4.5 features_manager/ — 特性管理系统

项目的**核心设计亮点**。采用声明式特性注册机制，将 50+ 种优化特性解耦为独立模块，按需组合注入。

```
features_manager/
├── __init__.py             # 特性列表初始化与注册
├── common/                 # 通用特性（数据/嵌入/训练默认值）
├── transformer/            # Transformer 相关特性
│   ├── flash_attention/    #   Flash Attention
│   ├── multi_latent_attention/  #   MLA
│   └── mtp.py              #   多 Token 预测
├── tensor_parallel/        # TP 特性
├── pipeline_parallel/      # PP 特性
├── context_parallel/       # CP 特性
├── moe/                    # MOE 特性
├── memory/                 # 显存优化特性
├── finetune/               # 微调特性（LoRA 等）
├── low_precision/          # 低精度优化
├── high_availability/      # 高可用特性
└── ...
```

---

## 5. 模块间关系

```
pretrain_gpt.py ──────► training/training.py::pretrain()
                              │
                              ├── features_manager/ ──► 注册并应用 50+ 特性
                              │       │
                              │       └── tasks/megatron_adaptor_v2.py ──► 适配到 Megatron
                              │
                              ├── tasks/models/ ──► 模型定义
                              ├── tasks/dataset/ ──► 数据加载
                              └── core/ ──► Transformer 构件 / 并行策略 / 优化器

train_fsdp2.py ──────► fsdp2/train/mindspeed_trainer.py
                              │
                              ├── fsdp2/models/ ──► 模型工厂 + 注册表
                              ├── fsdp2/data/ ──► 数据工厂
                              ├── fsdp2/optim/ ──► 优化器工厂
                              └── fsdp2/distributed/ ──► FSDP 分片与并行
```

---

## 6. 设计理念

### 6.1 面向昇腾硬件的全栈优化

项目不是通用 LLM 框架，而是**深度绑定华为 Ascend NPU** 的垂直解决方案。从算子融合（FusedRMSNorm、FusedSwiGLU）到通信优化（MC2、异步 AllReduce），所有优化路径都针对昇腾硬件特性设计。

### 6.2 双训练后端架构

提供两条并行的训练路径：
- **Megatron-core 路径**（成熟稳定）：基于 NVIDIA Megatron-LM 改造，支持最全的模型和特性。
- **FSDP2 路径**（新一代）：基于 PyTorch 原生 FSDP v2，架构更现代，采用工厂模式和注册表模式，正在逐步扩展模型支持。

### 6.3 特性即插件（Feature-as-Plugin）

`features_manager` 是项目最核心的架构模式。每个优化特性（Flash Attention、LoRA、MOE、TP/PP/CP 等）被封装为独立插件，通过统一的注册接口按需启用。这种设计：
- 避免了特性间的代码耦合
- 允许灵活组合不同的优化策略
- 降低了新增特性的开发成本

### 6.4 三层抽象架构

```
高层（What）：入口脚本 + 训练框架 → 定义"训练什么、怎么训练"
中层（How）：tasks/ → 实现"具体模型/数据/微调策略"
底层（With）：core/ → 提供"用什么组件构建"
正交（Which）：features_manager/ → 决定"启用哪些优化"
```

### 6.5 广泛的模型覆盖

以配置驱动的方式支持 100+ 种 LLM 架构，通过 `examples/` 下的 shell 脚本提供每个模型的标准训练配方，降低用户使用门槛。

### 6.6 全生命周期覆盖

不仅覆盖训练，还提供完整的模型开发工具链：
- **数据预处理** → **预训练** → **微调（SFT/LoRA/DPO/RLHF）** → **权重转换** → **推理** → **评估**

---

## 7. 支持的并行策略

| 策略 | 缩写 | 说明 |
|------|------|------|
| 数据并行 | DP | 数据分片到多卡 |
| 张量并行 | TP | 模型参数按张量维度切分（含 2D 变体） |
| 流水线并行 | PP | 模型按层切分到不同设备（含虚拟阶段、DualPipe） |
| 序列/上下文并行 | SP/CP | 长序列切分到多卡 |
| 专家并行 | EP | MOE 模型中专家分布到不同设备 |
| FSDP | FSDP | 全参数分片数据并行 |

支持上述策略的任意组合（如 TP + PP + DP + SP）。

---

## 8. 支持的训练范式

| 范式 | 说明 |
|------|------|
| 预训练 | 多样本预训练、Pack 预训练、从 HuggingFace 在线加载 |
| 全参微调 | 全量参数微调 |
| LoRA / QLoRA / LU-LoRA | 参数高效微调 |
| SFT | 有监督微调（多轮对话、多样本 Pack） |
| DPO | 直接偏好优化 |
| RLHF | 基于人类反馈的强化学习 |
| 推理 | 流式推理、对话推理 |
| 评估 | MMLU、C-Eval、BBH、HumanEval、GSM8K 等 8+ 基准 |

---

## 9. 技术栈与依赖

| 组件 | 版本 |
|------|------|
| Python | 3.x |
| PyTorch | 2.7.1 |
| torch_npu | 7.3.0 |
| Megatron-core | v0.12.1 |
| MindSpeed | 2.3.0 |
| Transformers (HF) | 4.57.1 |
| PEFT | 0.7.1 |
| CANN | 昇腾编译器 |

---

## 10. 项目统计

| 指标 | 数值 |
|------|------|
| Python 文件 | 410+ |
| 支持模型 | 100+ |
| 优化特性 | 50+ |
| 模型示例 | 32+ 目录 |
| 评估基准 | 8+ |
| 文档文件 | 60+ |
| 测试类别 | 5（ut/st/poc/coverage/pipeline） |
