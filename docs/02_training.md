# Qwen3 模型训练实战指南

## 1. 训练脚本总览

项目在 `examples/mcore/qwen3/` 下提供了 Qwen3 全系列模型的完整脚本：

| 类型 | 脚本 | 说明 |
|------|------|------|
| **预训练** | `pretrain_qwen3_8b_4K_ptd.sh` | 8B 模型预训练（本文重点） |
| **SFT 全参微调** | `tune_qwen3_8b_4K_full_ptd.sh` | 8B 模型全参数 SFT |
| **SFT LoRA 微调** | `tune_qwen3_8b_4K_lora_ptd.sh` | 8B 模型 LoRA SFT |
| 数据预处理（预训练） | `data_convert_qwen3_pretrain.sh` | 预训练数据转换 |
| 数据预处理（指令） | `data_convert_qwen3_instruction.sh` | SFT 指令数据转换 |
| 权重转换 | `ckpt_convert_qwen3_hf2mcore.sh` | HuggingFace → Megatron 格式 |
| 推理 | `generate_qwen3_8b_ptd.sh` | 8B 模型推理 |
| 评估 | `evaluate_qwen3_8b_ptd.sh` | 8B 模型评估 |

其他规模：`0.6b`、`1.7b`、`4b`、`14b`、`32b` 均有对应脚本。

---

## 2. 预训练启动脚本详解

以 `pretrain_qwen3_8b_4K_ptd.sh` 为例，完整脚本结构如下：

### 2.1 完整脚本

```bash
#!/bin/bash

# ==================== 环境变量 ====================
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0
export TASK_QUEUE_ENABLE=2

# ==================== 分布式配置 ====================
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# ==================== 路径配置（需要用户填写） ====================
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

# ==================== 并行与训练规模 ====================
TP=1
PP=2
CP=1
MBS=1
GBS=64
SEQ_LENGTH=4096
TRAIN_ITERS=2000

# ==================== 分布式启动参数 ====================
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ==================== 优化参数 ====================
OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-ascend-coc
"

# ==================== 训练超参数 ====================
TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH}
"

# ==================== 模型并行参数 ====================
MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP}
"

# ==================== 模型结构参数（Qwen3 8B） ====================
GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --qk-layernorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 36 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --norm-epsilon 1e-6
"

# ==================== 数据参数 ====================
DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

# ==================== 输出参数 ====================
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"

# ==================== 启动训练 ====================
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee logs/train_mcore_qwen3_8b.log
```

---

## 3. SFT 训练启动脚本详解

### 3.1 全参微调脚本（tune_qwen3_8b_4K_full_ptd.sh）

与预训练脚本的**关键差异**（仅列出不同部分）：

```bash
# 入口脚本不同
torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \    # ← 不是 pretrain_gpt.py

# 新增 TUNE_ARGS 参数组
TUNE_ARGS="
    --finetune \                    # 开启微调模式
    --stage sft \                   # 指定微调阶段为 SFT
    --is-instruction-dataset \      # 标记为指令数据集
    --prompt-type qwen3 \           # 使用 qwen3 对话模板
    --no-pad-to-seq-lengths         # 不填充到固定长度（节省计算）
"

# GBS 更小（16 vs 64），因为微调不需要太大 batch
GBS=16
```

### 3.2 LoRA 微调脚本（tune_qwen3_8b_4K_lora_ptd.sh）

在全参微调基础上**额外增加**的参数：

```bash
TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \       # 不使用 fast tokenizer
    --prompt-type qwen3 \
    --no-pad-to-seq-lengths \
    --lora-r 16 \                    # LoRA 秩
    --lora-alpha 32 \                # LoRA 缩放因子
    --lora-fusion \                  # 开启 LoRA 融合加速
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"                                    # LoRA 注入的目标模块

# MBS 更大（4 vs 1），因为 LoRA 显存占用更少
MBS=4
```

---

## 4. 环境变量详解

### 4.1 Ascend NPU 相关环境变量

| 环境变量 | 值 | 含义 |
|----------|-----|------|
| `HCCL_CONNECT_TIMEOUT` | `1800` | HCCL（华为集合通信库）连接超时时间（秒）。多节点训练时需设较大值，避免节点间建立连接超时 |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | 限制每个设备的最大并发 CUDA stream 连接数。设为 1 确保计算和通信的串行化，避免 NPU 上的资源竞争 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 显存分配策略。启用可扩展段式分配，减少显存碎片化，提升大模型训练的显存利用率 |
| `NPU_ASD_ENABLE` | `0` | 关闭 NPU 自动精度降级（Automatic precision downcast）。训练时需要精确控制精度，不需要自动降级 |
| `TASK_QUEUE_ENABLE` | `2` | NPU 任务队列模式。值为 2 启用高级任务调度，提升算子执行效率和 NPU 利用率 |

### 4.2 分布式训练环境变量

| 变量 | 含义 |
|------|------|
| `NPUS_PER_NODE` | 每节点 NPU 卡数（通常为 8） |
| `MASTER_ADDR` | 主节点 IP 地址（单机用 `localhost`） |
| `MASTER_PORT` | 主节点通信端口 |
| `NNODES` | 总节点数 |
| `NODE_RANK` | 当前节点编号（从 0 开始） |

---

## 5. 关键 --argument 参数分类详解

### 5.1 模型结构参数

| 参数 | Qwen3-8B 值 | 含义 |
|------|------------|------|
| `--use-mcore-models` | — | 使用 Megatron-core 模型实现（非 legacy） |
| `--spec` | `mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec` | 指定 Transformer 层的模块规格，定义了 Attention/MLP 的具体组件组合 |
| `--num-layers` | `36` | Transformer 层数 |
| `--hidden-size` | `4096` | 隐藏层维度 |
| `--ffn-hidden-size` | `12288` | FFN 中间层维度 |
| `--num-attention-heads` | `32` | 注意力头数 |
| `--num-query-groups` | `8` | GQA（Grouped Query Attention）的 KV 组数。8 组意味着每 4 个 Q head 共享 1 组 KV |
| `--group-query-attention` | — | 启用 GQA |
| `--qk-layernorm` | — | 在 Q/K 投影后应用 LayerNorm（Qwen3 特有） |
| `--max-position-embeddings` | `4096` | 最大位置编码长度 |
| `--padded-vocab-size` | `151936` | 词表大小（Qwen3 tokenizer） |
| `--rotary-base` | `1000000` | RoPE 旋转位置编码的基频 |
| `--position-embedding-type` | `rope` | 位置编码类型 |
| `--normalization` | `RMSNorm` | 归一化方法 |
| `--norm-epsilon` | `1e-6` | 归一化 epsilon |
| `--swiglu` | — | 使用 SwiGLU 激活函数 |
| `--disable-bias-linear` | — | 线性层不使用偏置（Qwen3 设计） |
| `--untie-embeddings-and-output-weights` | — | 输入嵌入和输出投影不共享权重 |

### 5.2 并行策略参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `--tensor-model-parallel-size` | `1` | 张量并行度。模型参数按张量维度切分到多卡 |
| `--pipeline-model-parallel-size` | `2` | 流水线并行度。模型按层切分到多组设备 |
| `--sequence-parallel` | — | 启用序列并行（SFT 脚本中开启），在 TP 组内按序列维度切分 |
| `--context-parallel-size` | `1`（默认） | 上下文并行度，用于超长序列训练 |

**DP 自动计算**：`DP = WORLD_SIZE / (TP × PP)`，8 卡 TP=1 PP=2 → DP=4

### 5.3 训练超参数

| 参数 | 预训练值 | 含义 |
|------|---------|------|
| `--micro-batch-size` | `1` | 每个设备每个 micro-batch 的样本数 |
| `--global-batch-size` | `64` | 全局 batch size = MBS × DP × 梯度累积步数 |
| `--lr` | `1.25e-6` | 学习率（预训练较小因为是续训） |
| `--lr-decay-style` | `cosine` | 学习率衰减策略 |
| `--min-lr` | `1.25e-7` | 最小学习率 |
| `--lr-warmup-fraction` | `0.01` | warmup 占总步数的比例 |
| `--weight-decay` | `0.1` | 权重衰减 |
| `--clip-grad` | `1.0` | 梯度裁剪阈值 |
| `--adam-beta1` / `--adam-beta2` | `0.9` / `0.95` | Adam 优化器参数 |
| `--initial-loss-scale` | `4096` | BF16 混合精度训练的初始 loss scale |
| `--bf16` | — | 使用 BF16 精度训练 |
| `--train-iters` | `2000` | 总训练步数 |
| `--seq-length` | `4096` | 训练序列长度 |
| `--seed` | `42` | 随机种子 |

### 5.4 优化加速参数

| 参数 | 含义 |
|------|------|
| `--use-flash-attn` | 使用 Flash Attention 加速注意力计算 |
| `--use-fused-rotary-pos-emb` | 使用融合版旋转位置编码算子 |
| `--use-fused-swiglu` | 使用融合版 SwiGLU 算子 |
| `--use-fused-rmsnorm` | 使用融合版 RMSNorm 算子 |
| `--no-masked-softmax-fusion` | 禁用 masked softmax 融合（NPU 上不适用） |
| `--use-distributed-optimizer` | 使用分布式优化器（将优化器状态分片到各 DP rank，节省显存） |
| `--reuse-fp32-param` | 复用 FP32 参数副本，进一步节省优化器显存 |
| `--overlap-grad-reduce` | 梯度 AllReduce 与反向传播重叠（通信隐藏） |
| `--overlap-param-gather` | 参数 AllGather 与前向传播重叠 |
| `--use-ascend-coc` | 使用昇腾 CoC（Communication over Computation）计算通信融合 |
| `--no-gradient-accumulation-fusion` | 禁用梯度累积融合（NPU 上有时更稳定） |
| `--attention-softmax-in-fp32` | Softmax 在 FP32 精度下计算（数值稳定性） |

### 5.5 SFT 微调专属参数

| 参数 | 含义 |
|------|------|
| `--finetune` | 开启微调模式（加载预训练权重后重置训练状态） |
| `--stage sft` | 指定训练阶段为 SFT |
| `--is-instruction-dataset` | 标记数据为指令格式（影响 get_batch 和 loss 计算） |
| `--prompt-type qwen3` | 使用 Qwen3 对话模板格式化数据 |
| `--no-pad-to-seq-lengths` | 不将样本填充到 seq_length，按实际长度计算（节省算力） |

### 5.6 LoRA 专属参数

| 参数 | 含义 |
|------|------|
| `--lora-r 16` | LoRA 低秩矩阵的秩（rank），越大表达能力越强但参数越多 |
| `--lora-alpha 32` | LoRA 缩放因子，通常设为 `2 × lora-r` |
| `--lora-fusion` | 启用 LoRA 计算融合加速 |
| `--lora-target-modules` | LoRA 注入的目标模块：`linear_qkv`（QKV 投影）、`linear_proj`（输出投影）、`linear_fc1`/`linear_fc2`（FFN） |

### 5.7 数据与 I/O 参数

| 参数 | 含义 |
|------|------|
| `--data-path` | 预处理后的数据路径前缀（不含 `.bin`/`.idx` 后缀） |
| `--split 100,0,0` | train/valid/test 的比例切分 |
| `--tokenizer-type PretrainedFromHF` | 使用 HuggingFace 预训练 tokenizer |
| `--tokenizer-name-or-path` | HuggingFace tokenizer 路径 |
| `--load` | 加载检查点的路径 |
| `--save` | 保存检查点的路径 |
| `--log-interval` | 日志输出间隔（步数） |
| `--save-interval` | 检查点保存间隔 |
| `--eval-interval` | 评估间隔 |
| `--no-load-optim` | 不加载优化器状态（从头训练或微调时使用） |
| `--no-load-rng` | 不加载随机状态 |
| `--distributed-backend nccl` | 分布式后端（NPU 上实际由 HCCL 驱动，但接口兼容 NCCL） |
| `--transformer-impl local` | 使用本地 Transformer 实现（非 TransformerEngine） |

---

## 6. 数据集格式

### 6.1 预训练数据格式

**原始输入格式**（支持 `.parquet`、`.json`、`.jsonl`、`.csv`、`.txt`）：

```json
{"text": "这是一段训练文本。模型会学习预测下一个 token。"}
{"text": "另一段训练文本。每行一个文档。"}
```

**预处理命令**（`data_convert_qwen3_pretrain.sh`）：

```bash
python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf/ \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ./dataset/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000
```

**处理流程**：
```
原始文本 → 分句 → Tokenize → 追加 EOD token → 写入二进制文件
```

**输出文件**：
```
./dataset/enwiki_text_document.bin   # 二进制 token 数据
./dataset/enwiki_text_document.idx   # 索引文件
```

**在训练脚本中引用**（不含后缀）：
```bash
DATA_PATH="./dataset/enwiki_text_document"
```

### 6.2 SFT 指令数据格式

**原始输入格式（Alpaca 格式）**：

```json
[
  {
    "instruction": "请解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进...",
    "system": "你是一个乐于助人的 AI 助手。",
    "history": [
      ["你好", "你好！有什么可以帮助你的吗？"]
    ]
  }
]
```

| 字段 | 必填 | 含义 |
|------|------|------|
| `instruction` | 是 | 用户指令 |
| `input` | 否 | 用户额外输入（上下文） |
| `output` | 是 | 期望的模型回答 |
| `system` | 否 | 系统提示词 |
| `history` | 否 | 历史多轮对话 `[["用户", "助手"], ...]` |

**预处理命令**（`data_convert_qwen3_instruction.sh`）：

```bash
python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --enable-thinking true \
    --prompt-type qwen3
```

**处理流程**：
```
Alpaca JSON → 按 qwen3 模板格式化 → Tokenize → 生成 labels（instruction 部分标记为 -100）→ 写入二进制
```

**qwen3 对话模板**（来自 `configs/finetune/templates.json`）：

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

**输出文件**（三个字段各一组）：
```
./finetune_dataset/alpaca_packed_input_ids_document.bin/.idx
./finetune_dataset/alpaca_packed_labels_document.bin/.idx
./finetune_dataset/alpaca_packed_attention_mask_document.bin/.idx
```

**在训练脚本中引用**：
```bash
DATA_PATH="./finetune_dataset/alpaca_packed"
```

### 6.3 ShareGPT 多轮对话格式（补充）

```json
[
  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！有什么可以帮助你的吗？"},
      {"from": "human", "value": "解释一下量子计算"},
      {"from": "gpt", "value": "量子计算是利用量子力学原理进行计算的..."}
    ],
    "system": "你是一个专业的科学助手。"
  }
]
```

使用 `ShareGPTInstructionHandler` 进行预处理。

---

## 7. 单机 8 卡 NPU 运行指南

### 7.1 前置准备清单

```
1. 安装昇腾驱动和 CANN toolkit
   source /usr/local/Ascend/ascend-toolkit/set_env.sh

2. 安装 Python 依赖
   pip install -r requirements.txt

3. 安装 MindSpeed（昇腾优化库）
   参考 docs/pytorch/install_guide.md

4. 下载 Qwen3 HuggingFace 权重
   目录结构：model_from_hf/qwen3_hf/
   ├── config.json
   ├── tokenizer.json
   ├── model-00001-of-000XX.safetensors
   └── ...

5. 转换权重：HuggingFace → Megatron 格式
   bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh

6. 准备并预处理数据
   bash examples/mcore/qwen3/data_convert_qwen3_pretrain.sh      # 预训练
   bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh   # SFT
```

### 7.2 需要修改的参数

#### 必须修改的路径参数

```bash
# 4 个路径全部替换为真实路径
CKPT_LOAD_DIR="/path/to/qwen3_mcore_weights"    # Megatron 格式权重目录
CKPT_SAVE_DIR="/path/to/save/checkpoints"        # 检查点保存目录
DATA_PATH="/path/to/dataset/enwiki_text_document" # 预处理后数据路径（不含.bin/.idx）
TOKENIZER_PATH="/path/to/qwen3_hf"               # HuggingFace tokenizer 目录
```

#### 分布式配置（单机 8 卡无需改动）

```bash
NPUS_PER_NODE=8        # 已正确
MASTER_ADDR=localhost   # 单机用 localhost
MASTER_PORT=6000        # 确保端口未被占用
NNODES=1                # 单机
NODE_RANK=0             # 单机
```

#### 并行策略调优建议

**Qwen3-8B（单机 8 卡推荐配置）**：

| 场景 | TP | PP | DP（自动） | MBS | GBS | 说明 |
|------|----|----|-----------|-----|-----|------|
| **预训练** | 1 | 2 | 4 | 1 | 64 | 原始脚本默认值，推荐直接使用 |
| **SFT 全参** | 1 | 2 | 4 | 1 | 16 | 原始脚本默认值 |
| **SFT LoRA** | 1 | 2 | 4 | 4 | 16 | 显存占用更少，MBS 可加大 |
| 备选方案 | 2 | 1 | 4 | 2 | 64 | 如果遇到 PP 通信瓶颈 |
| 显存紧张 | 2 | 2 | 2 | 1 | 32 | 牺牲 DP 换取更低显存 |

> **约束**：`TP × PP × DP = WORLD_SIZE`（8），`GBS` 必须能被 `MBS × DP` 整除

**Qwen3-32B（单机 8 卡推荐配置）**：

| 场景 | TP | PP | DP | MBS | GBS |
|------|----|----|-----|-----|-----|
| 预训练 | 4 | 2 | 1 | 1 | 32 |
| SFT LoRA | 2 | 2 | 2 | 1 | 8 |

> 32B 模型在单机 8 卡上 DP 通常只能为 1-2，受显存限制

#### 显存不足时的缓解策略

按优先级从高到低：

```bash
# 1. 降低 micro-batch-size（最直接）
--micro-batch-size 1

# 2. 启用重计算（用计算换显存）
--recompute-method block \
--recompute-granularity full \
--recompute-num-layers 36        # 全部层都重计算

# 3. 降低序列长度
--seq-length 2048

# 4. 使用分布式优化器（已默认开启）
--use-distributed-optimizer

# 5. 增大 TP 或 PP
TP=2  PP=2  # 将模型切到更多卡上
```

### 7.3 权重转换参数调整

`ckpt_convert_qwen3_hf2mcore.sh` 中的并行度必须与训练脚本一致：

```bash
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \        # ← 与训练脚本的 TP 一致
    --target-pipeline-parallel-size 2 \      # ← 与训练脚本的 PP 一致
    --load-dir ./model_from_hf/qwen3_hf/ \
    --save-dir ./model_weights/qwen3_mcore/ \
    --model-type-hf qwen3
```

### 7.4 快速验证运行

建议先用小参数做冒烟测试：

```bash
# 修改训练脚本中的以下参数
TRAIN_ITERS=10       # 只训练 10 步
GBS=8                # 小 batch
SEQ_LENGTH=512       # 短序列
--eval-iters 0       # 不评估
--save-interval 10   # 最后保存一次
```

### 7.5 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `HCCL timeout` | 节点间通信超时 | 增大 `HCCL_CONNECT_TIMEOUT`，检查网络 |
| `OOM (Out of Memory)` | NPU 显存不足 | 减小 MBS、启用重计算、增大 TP/PP |
| `Checkpoint load failed` | TP/PP 与权重不匹配 | 重新执行 `convert_ckpt_v2.py`，保持并行度一致 |
| `Data path not found` | 数据路径错误 | 检查 DATA_PATH 是否指向 `.bin`/`.idx` 的前缀（不含后缀） |
| `Tokenizer not found` | tokenizer 路径错误 | 确保路径指向含 `tokenizer.json` 的目录 |
| `Port already in use` | 端口冲突 | 修改 `MASTER_PORT` 为其他值 |
| `GBS not divisible` | GBS/(MBS×DP) 不是整数 | 调整 GBS 使其能被 MBS×DP 整除 |

### 7.6 日志输出目录

脚本默认将日志输出到 `logs/` 目录，运行前需创建：

```bash
mkdir -p logs
```

---

## 8. 端到端操作示例

### 8.1 预训练端到端流程

```bash
# Step 1: 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p logs dataset model_weights

# Step 2: 数据预处理
python ./preprocess_data.py \
    --input ./raw_data/my_corpus.jsonl \
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf/ \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ./dataset/my_corpus \
    --json-keys text \
    --workers 4 \
    --log-interval 1000

# Step 3: 权重转换（TP=1, PP=2）
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/qwen3_hf/ \
    --save-dir ./model_weights/qwen3_mcore/ \
    --model-type-hf qwen3

# Step 4: 修改脚本路径后启动训练
# 编辑 examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh 中的 4 个路径
bash examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh
```

### 8.2 SFT 微调端到端流程

```bash
# Step 1: 准备 Alpaca 格式指令数据（my_sft_data.json）
cat > ./raw_data/my_sft_data.json << 'EOF'
[
  {
    "instruction": "请总结以下文章的要点",
    "input": "人工智能（AI）正在改变各行各业...",
    "output": "文章主要讨论了以下几个要点：1. AI 在各行业的应用...",
    "system": "",
    "history": []
  }
]
EOF

# Step 2: 数据预处理
python ./preprocess_data.py \
    --input ./raw_data/my_sft_data.json \
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf/ \
    --output-prefix ./finetune_dataset/my_sft \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen3

# Step 3: 修改脚本路径后启动 SFT
# DATA_PATH="./finetune_dataset/my_sft_packed"
# CKPT_LOAD_DIR="./model_weights/qwen3_mcore/"
bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
```
