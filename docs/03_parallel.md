# 模型并行与分布式训练实现分析

## 1. 并行策略总览

MindSpeed-LLM 支持 6 种并行策略的任意组合：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WORLD_SIZE = TP × PP × CP × DP                  │
│                                                                     │
│  TP（张量并行）    模型参数按张量维度切分到多卡                       │
│  PP（流水线并行）  模型按层切分到多组设备                             │
│  CP（上下文并行）  长序列按序列维度切分到多卡                         │
│  DP（数据并行）    数据分片到多组（自动计算）                         │
│  EP（专家并行）    MOE 专家分布到不同设备（从 DP 中划分）             │
│  SP（序列并行）    TP 组内按序列维度切分激活值（TP 的伴生策略）       │
└─────────────────────────────────────────────────────────────────────┘
```

**约束关系**：
```
DP = WORLD_SIZE / (TP × PP × CP)
EP 从 DP×CP 中划分：(DP × CP) % EP == 0
```

---

## 2. 配置入口：从命令行到进程组

### 2.1 命令行参数

```bash
# 核心并行参数
--tensor-model-parallel-size 4      # TP
--pipeline-model-parallel-size 2     # PP
--context-parallel-size 8            # CP
--expert-model-parallel-size 4       # EP（MOE 模型）
--sequence-parallel                  # SP（开关）

# 虚拟流水线
--num-layers-per-virtual-pipeline-stage 4  # VPP

# 2D 张量并行
--tp-2d                              # 开启 2D TP
--tp-x 2                             # TP X 维度
--tp-y 4                             # TP Y 维度（TP = tp_x × tp_y）

# 上下文并行算法
--context-parallel-algo ulysses_cp_algo   # Ulysses 或 megatron_cp_algo (Ring)
```

### 2.2 参数解析与验证

```
training/arguments.py::parse_args()
  │
  ├── Megatron 原生参数：TP, PP, DP, SP
  ├── MindSpeed-LLM 扩展参数：EP, CP algo, 2D TP, VPP
  └── validate_args()  ← 校验并行度之间的整除关系
```

### 2.3 进程组初始化

所有并行通信组在 `initialize_megatron()` → `_initialize_distributed()` → `initialize_model_parallel()` 中创建。

MindSpeed-LLM 通过装饰器 `initialize_model_parallel_decorator` 扩展了 Megatron 原生的初始化逻辑。

**关键代码**：`mindspeed_llm/core/parallel_state.py:35`

```python
def initialize_model_parallel_decorator(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        order="tp-cp-ep-dp-pp",     # ← 进程组排布顺序
        ...
    ):
        # 1. 调用 Megatron 原生初始化（创建 TP/PP/DP 基础进程组）
        initialize_model_parallel(...)

        # 2. MindSpeed-LLM 扩展：重建 EP 相关进程组
        #    因为 Megatron 原生将 EP 设为 1，这里按实际 EP 重新划分
        for dp_cp_ranks in all_data_parallel_group_ranks_with_cp:
            for i in range(0, len(dp_cp_ranks), expert_model_parallel_size):
                ranks = dp_cp_ranks[i:i + expert_model_parallel_size]
                group = torch.distributed.new_group(ranks, ...)
                # → _EXPERT_MODEL_PARALLEL_GROUP

        # 3. 初始化上下文并行的 send/recv overlap 组
        initialize_context_parallel_group_for_send_recv_overlap(...)
        initialize_context_parallel_group_for_hybrid_cp(...)
        initialize_context_parallel_group_for_double_ring(...)

        # 4. 初始化 2D 张量并行进程组（如果启用）
        if args.tp_2d:
            TensorParallelYUnionCP(parallel_cfg=SimpleParallelCfg(
                dp, pp, tp, cp, ep, tp_x, tp_y
            ))
```

---

## 3. 进程组排布详解

### 3.1 排布顺序：`tp-cp-ep-dp-pp`

默认排布下，对于 16 个 GPU（TP=2, PP=2, CP=2, DP=2）：

```
Rank:  0  1 | 2  3 | 4  5 | 6  7 | 8  9 | 10 11 | 12 13 | 14 15
       ├TP┤   ├TP┤   ├TP┤   ├TP┤   ├TP┤    ├TP┤    ├TP┤    ├TP┤
       ├──CP──┤       ├──CP──┤       ├──CP──┤        ├──CP──┤
       ├──────DP──────┤              ├──────DP───────┤
       ├──────────────PP Stage 0──────────────────────────────────┤
                                     ├──────────PP Stage 1────────┤
```

### 3.2 创建的进程组类型

| 进程组 | 全局变量 | 用途 |
|--------|----------|------|
| `_TENSOR_MODEL_PARALLEL_GROUP` | TP 组 | AllReduce / AllGather 参数和梯度 |
| `_PIPELINE_MODEL_PARALLEL_GROUP` | PP 组 | P2P Send/Recv 激活值 |
| `_DATA_PARALLEL_GROUP` | DP 组 | AllReduce 梯度 |
| `_CONTEXT_PARALLEL_GROUP` | CP 组 | AllGather / ReduceScatter 序列块 |
| `_EXPERT_MODEL_PARALLEL_GROUP` | EP 组 | AlltoAll 分发 token 到专家 |
| `_DATA_MODULO_EXPERT_PARALLEL_GROUP` | DP%EP 组 | 非专家参数的 DP 梯度同步 |
| `_TENSOR_AND_EXPERT_PARALLEL_GROUP` | TP×EP 组 | 专家层的张量+专家联合并行 |

---

## 4. 张量并行（Tensor Parallelism）实现

### 4.1 核心原理

将矩阵乘法按行或列维度切分到多个设备上并行计算。

```
标准线性层：Y = XW + b

列并行（ColumnParallelLinear）：
  W 按列切分：W = [W1 | W2]
  每个 rank 计算：Yi = X @ Wi
  输出拼接或独立使用

行并行（RowParallelLinear）：
  W 按行切分：W = [W1; W2]
  输入切分：X = [X1 | X2]
  每个 rank 计算：Yi = Xi @ Wi
  输出 AllReduce：Y = sum(Yi)
```

### 4.2 在 Transformer 中的应用

**关键代码**：Attention 层和 MLP 层中的 TP 切分

```python
# Qwen3 Layer Spec (tasks/models/spec/qwen3_spec.py)
layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=SelfAttention,
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,    # ← QKV 投影：列并行
                core_attention=DotProductAttention,  # ← 注意力计算：本地
                linear_proj=RowParallelLinear,       # ← 输出投影：行并行
            ),
        ),
        mlp=ModuleSpec(
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,     # ← FFN 上投影：列并行
                linear_fc2=RowParallelLinear,        # ← FFN 下投影：行并行
            ),
        ),
    ),
)
```

**通信模式**：

```
前向传播：
  输入 X（所有 rank 相同）
    → ColumnParallel: 各 rank 独立计算，无通信
    → Attention / SwiGLU: 各 rank 独立计算
    → RowParallel: AllReduce 输出
    → 输出 Y（所有 rank 相同）

反向传播：
  梯度通过对称的 AllReduce 反向传播
```

### 4.3 词嵌入的张量并行

**关键代码**：`core/tensor_parallel/layers.py:37`

```python
def vocab_embedding_init_func(self, num_embeddings, embedding_dim, ...):
    # 按词表维度切分到各 TP rank
    (self.vocab_start_index, self.vocab_end_index) = \
        VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            parallel_state.get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size
        )
    self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
    # 每个 rank 只存储 vocab_size / TP 大小的嵌入表
```

### 4.4 序列并行（SP）

当 `--sequence-parallel` 开启时，TP 组内的激活值按序列维度切分，减少显存占用：

```
无 SP：每个 TP rank 存储完整序列长度的激活值
有 SP：每个 TP rank 只存储 seq_length / TP 的激活值

ColumnParallel 前：AllGather 收集完整序列
RowParallel 后：ReduceScatter 切分序列

通信量不变，但激活值显存减少 TP 倍
```

### 4.5 2D 张量并行

**关键代码**：`core/transformer/attention.py:22`

当 `--tp-2d` 启用时，TP 组进一步拆分为 `tp_x × tp_y` 的二维网格：

```python
def self_attention_init_tp2d_wrapper(fn):
    def wrapper(self, config, submodules, layer_number, ...):
        fn(self, config, submodules, layer_number, ...)
        if args.tp_2d:
            # QKV 投影：沿 tp_x 维度 AllGather 输入，沿 tp_y 维度 ReduceScatter 输出
            self.linear_qkv = ParallelLinear2D(
                ag_comm_intf=TPXCollectiveComm,      # AllGather 沿 X 维度
                rs_comm_intf=TPYCollectiveComm,       # ReduceScatter 沿 Y 维度
                ...
            )
            # 输出投影：方向相反
            self.linear_proj = ParallelLinear2D(
                ag_comm_intf=TPYCollectiveComm,       # AllGather 沿 Y 维度
                rs_comm_intf=TPXCollectiveComm,       # ReduceScatter 沿 X 维度
                ...
            )
```

**优势**：减少单次通信的数据量，将大的 AllReduce 拆分为两次较小的通信。

---

## 5. 流水线并行（Pipeline Parallelism）实现

### 5.1 层分配

**关键代码**：`core/transformer/transformer_block.py:48`

```python
def get_num_layers_to_build(config: TransformerConfig) -> int:
    # 每个 PP 阶段分配的层数
    num_layers_per_pipeline_rank = (
        config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
    )

    # 虚拟流水线：进一步按 VPP 切分
    if vp_size is not None:
        num_layers_to_build = num_layers_per_pipeline_rank // vp_size
    else:
        num_layers_to_build = num_layers_per_pipeline_rank

    # 支持不均匀层分配
    if config.num_layer_list:
        pp_stage = parallel_state.get_pipeline_model_parallel_rank()
        num_layers_to_build = num_layer_list[pp_stage]

    return num_layers_to_build
```

**示例：64 层模型，PP=2，VPP=4**

```
Stage 0 的虚拟阶段分配：
  Chunk 0: 层 [0-7]     ← 前向先执行
  Chunk 1: 层 [16-23]
  Chunk 2: 层 [32-39]
  Chunk 3: 层 [48-55]

Stage 1 的虚拟阶段分配：
  Chunk 0: 层 [8-15]
  Chunk 1: 层 [24-31]
  Chunk 2: 层 [40-47]
  Chunk 3: 层 [56-63]   ← 最后阶段计算 loss
```

### 5.2 调度策略

**关键代码**：`core/pipeline_parallel/schedules.py`

训练循环中 `train_step()` 调用 `get_forward_backward_func()` 选择调度策略：

```python
def get_forward_backward_func_wrapper(get_forward_backward_func):
    def wrapper(*args, **kwargs):
        forward_backward_func = get_forward_backward_func(*args, **kwargs)
        # 如果启用 RiPipe，替换为 RiPipe 调度
        if arguments.recompute_in_advance and torch.is_grad_enabled():
            forward_backward_func = forward_backward_ripipe_pipelining
        return forward_backward_func
```

#### 无流水线（PP=1）

```
micro-batch 0:  [Forward] → [Backward]
micro-batch 1:  [Forward] → [Backward]
...
AllReduce 梯度 → optimizer.step()
```

#### 1F1B 调度（PP>1, 无 VPP）

```
时间 →
Stage 0: [F0][F1][F2][F3] [B3][F4][B2][F5][B1][F6][B0]...
Stage 1:     [F0][F1][F2] [B3][F3][B2][F4][B1][F5][B0]...

F = Forward, B = Backward, 数字 = micro-batch ID
```

**通信**：Stage 之间通过 P2P Send/Recv 传递激活值和梯度。

#### 交错式调度（PP>1, 有 VPP）

```
每个 Stage 有 VPP 个模型块，交替执行：
Stage 0: [F0_c0][F0_c1][F1_c0][F1_c1]...[B3_c1][B3_c0]...

c0 = chunk 0, c1 = chunk 1
```

**优势**：减少流水线气泡（bubble），提升 GPU 利用率。

#### DualPipe 调度

**关键代码**：`core/pipeline_parallel/dualpipe/`

双向流水线，前向从两端同时开始，进一步减少气泡。

### 5.3 P2P 通信

PP 阶段间通过 `torch.distributed.send()` / `recv()` 传递数据：

```
Stage 0 → Stage 1：发送前向激活值（forward pass）
Stage 1 → Stage 0：发送反向梯度（backward pass）
```

MindSpeed-LLM 对 P2P 通信进行了优化（`OptimizeP2PCommFeature`、`OptimizeSendRecvCommFeature`）。

---

## 6. 上下文并行（Context Parallelism）实现

### 6.1 核心思想

将长序列按序列维度切分到 CP 组内的多个设备，每个设备只处理序列的一个分块。

### 6.2 两种算法

#### Ring Attention（`megatron_cp_algo`）

```
CP Rank 0: [token 0..S/4]   ←─ Ring 传递 KV ──→  CP Rank 1: [token S/4..S/2]
                                                          ↕
CP Rank 3: [token 3S/4..S]  ←─ Ring 传递 KV ──→  CP Rank 2: [token S/2..3S/4]
```

每个 rank 持有 Q 的一个分块，KV 通过 Ring 通信循环传递，逐步累积注意力结果。

#### Ulysses（`ulysses_cp_algo`）

```
1. AlltoAll：将 [batch, seq/CP, heads, dim] 转为 [batch, seq, heads/CP, dim]
2. 各 rank 独立计算 heads/CP 个头的注意力
3. AlltoAll：将结果转回 [batch, seq/CP, heads, dim]
```

**适用条件**：`num_attention_heads % CP == 0`

### 6.3 数据分发

**关键代码**：`pretrain_gpt.py:132`

```python
# 在 get_batch() 中按 CP rank 切分 batch
batch = get_batch_on_this_cp_rank(batch)
# 将 tokens/labels/loss_mask 等按序列维度切分
```

---

## 7. 数据并行（Data Parallelism）实现

### 7.1 标准 DDP

每个 DP rank 处理不同的数据分片，前向/反向独立计算，然后 AllReduce 梯度：

```python
# Megatron DDP 封装
from megatron.core.distributed import DistributedDataParallel as DDP

# 梯度同步
config.finalize_model_grads_func = finalize_model_grads  # AllReduce 跨 DP 组
```

### 7.2 分布式优化器

`--use-distributed-optimizer` 将优化器状态分片到各 DP rank：

```
标准 DDP：每个 rank 存储完整的优化器状态（参数 + momentum + variance）
分布式优化器：每个 rank 只存储 1/DP 的优化器状态

通信变化：
  梯度同步：AllReduce → ReduceScatter（各 rank 只收集自己负责的参数梯度）
  参数广播：无 → AllGather（更新后广播最新参数）
```

**显存节省**：Adam 优化器的状态从 12 × params 降为 12 × params / DP。

### 7.3 通信重叠

```bash
--overlap-grad-reduce     # 梯度 ReduceScatter 与反向传播重叠
--overlap-param-gather    # 参数 AllGather 与前向传播重叠
```

**关键代码**：`core/distributed/param_and_grad_buffer.py`

```python
def start_grad_sync_wrapper(fn):
    # 在反向传播计算的同时，异步启动梯度同步通信
    # 当一个 bucket 的梯度计算完成时，立即启动 AllReduce/ReduceScatter
```

---

## 8. 专家并行（Expert Parallelism）实现

### 8.1 核心思想

MOE 模型中，不同专家分布到不同设备。Token 通过 Router 分配到对应专家所在的设备。

### 8.2 进程组创建

```python
# parallel_state.py: 从 DP×CP 中划分 EP 组
for dp_cp_ranks in all_data_parallel_group_ranks_with_cp:
    for i in range(0, len(dp_cp_ranks), expert_model_parallel_size):
        ranks = dp_cp_ranks[i:i + expert_model_parallel_size]
        group = torch.distributed.new_group(ranks)
        # → _EXPERT_MODEL_PARALLEL_GROUP
```

### 8.3 Router 与 AlltoAll

**关键代码**：`core/transformer/moe/router.py`

```python
def group_limited_greedy_topKgating(self, logits):
    # 1. 计算每个 token 对每个专家的分数
    scores = F.softmax(logits, dim=1)
    # 2. 按组选 topk 组
    group_idx = torch.topk(group_scores, k=self.topk_group, ...)
    # 3. 在选中的组内选 topk 专家
    topk_weight, topk_idx = torch.topk(tmp_scores, k=moe_router_topk, ...)
    # 4. AlltoAll 将 token 分发到对应专家所在的 EP rank
```

**通信模式**：
```
Token → Router → [AlltoAll 分发] → 各 EP rank 的专家 → [AlltoAll 收集] → 输出
```

---

## 9. FSDP2 路径的并行实现

### 9.1 并行引擎

**关键代码**：`fsdp2/distributed/mindspeed_parallel_engine.py:20`

```python
class MindSpeedParallelEngine(torch.nn.Module):
    def __init__(self, config: ParallelEngineConfig, model, ...):
        self.parallel_state = init_parallel_state(config)
        self.apply_tp_modules()        # 张量并行
        self.apply_ep_modules()        # 专家并行
        self.apply_cp_modules()        # 上下文并行
        self.apply_recompute_modules() # 重计算
        self.apply_quantization_modules()  # 量化
        self.apply_fsdp_modules()      # FSDP 分片（最后应用）
```

**应用顺序严格**：TP → EP → CP → Recompute → FSDP，确保参数先做并行切分，再做 FSDP 分片。

### 9.2 配置声明

FSDP2 路径使用 `ParallelEngineConfig` 声明式配置：

```python
@dataclass
class ParallelEngineConfig:
    tensor_parallel_size: int = 1
    tp_plan: TPPlanConfig = None        # 指定哪些模块做列/行并行

    fully_shard_parallel_size: int = 1
    fsdp_plan: FSDPPlanConfig = None    # 指定 FSDP 分片粒度

    context_parallel_size: int = 1
    cp_plan: CPPlanConfig = None        # CP 配置

    expert_parallel_size: int = 1
    ep_plan: EPPlanConfig = None        # EP 配置

    recompute: bool = False
    recompute_plan: List[str] = None    # 指定哪些模块做重计算
```

```python
# TP Plan 示例
TPPlanConfig(
    colwise_parallel=["linear_qkv", "linear_fc1"],  # 列并行模块名
    rowwise_parallel=["linear_proj", "linear_fc2"],  # 行并行模块名
    sequence_parallel=["input_layernorm", "..."],     # SP 模块名
)
```

### 9.3 与 Megatron 路径的差异

| 维度 | Megatron 路径 | FSDP2 路径 |
|------|--------------|------------|
| TP 实现 | `ColumnParallelLinear` / `RowParallelLinear`（模型内置） | `DTensor` + `tensor_parallel_modules()`（模型外部包装） |
| PP 实现 | Megatron P2P 调度（1F1B / 交错 / DualPipe） | 暂不支持 PP |
| DP 实现 | Megatron DDP + 分布式优化器 | PyTorch FSDP2（原生全参数分片） |
| CP 实现 | Ring / Ulysses / Hybrid | Ulysses |

---

## 10. 多机多卡分布式训练指南

### 10.1 集群网络要求

```
节点间：RDMA 高速网络（推荐 200Gbps+ InfiniBand 或 RoCE）
节点内：HCCS（华为芯片间高速互联）/ PCIe
通信库：HCCL（华为集合通信库），接口兼容 NCCL
```

### 10.2 多机启动方式

使用 `torchrun` 在每个节点上分别启动：

```bash
# ========== 节点 0（主节点）==========
NNODES=4                     # 总节点数
NODE_RANK=0                  # 当前节点编号
MASTER_ADDR=192.168.1.100    # 主节点 IP（所有节点填相同值）
MASTER_PORT=6000             # 主节点端口

torchrun \
    --nproc_per_node 8 \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    pretrain_gpt.py ...

# ========== 节点 1 ==========
NODE_RANK=1
MASTER_ADDR=192.168.1.100    # 同上
torchrun ... --node_rank $NODE_RANK ...

# ========== 节点 2 ==========
NODE_RANK=2
...

# ========== 节点 3 ==========
NODE_RANK=3
...
```

**实际脚本中需要修改的变量**：

```bash
NPUS_PER_NODE=8               # 每节点 NPU 数
NNODES=4                      # 总节点数
NODE_RANK=0                   # 每个节点不同（0, 1, 2, 3）
MASTER_ADDR=192.168.1.100     # 主节点真实 IP（非 localhost）
MASTER_PORT=6000              # 所有节点一致
```

### 10.3 多机环境变量

```bash
# 基础
export HCCL_CONNECT_TIMEOUT=1800      # 多机时适当增大（秒）
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 多机网络专用
export HCCL_EXEC_TIMEOUT=5400         # 操作执行超时
export HCCL_IF_BASE_PORT=48890        # HCCL 通信基端口
export CPU_AFFINITY_CONF=1            # CPU 亲和性绑定（提升 NUMA 性能）

# 可选：指定网络接口
export GLOO_SOCKET_IFNAME=eth0        # Gloo 后端网卡
export HCCL_IF_NAME=eth0              # HCCL 通信网卡
```

### 10.4 并行策略规划

#### 原则

1. **TP 优先在节点内**：TP 通信量大且频繁，必须放在节点内高速互联上
2. **PP 可跨节点**：PP 只有 P2P 点对点通信，带宽需求较低
3. **DP 可跨节点**：DP 的 AllReduce 通信量大但频率低（每步一次）
4. **CP 尽量在节点内**：CP 的 Ring/AlltoAll 通信量较大

#### 典型集群配置方案

**场景 A：Qwen3-8B，4 节点 × 8 卡 = 32 卡**

```bash
TP=1  PP=1  DP=32  # 8B 模型单卡可放下，纯数据并行吞吐最高
GBS=256  MBS=2
```

**场景 B：Qwen3-32B，4 节点 × 8 卡 = 32 卡**

```bash
TP=4  PP=2  DP=4   # 32B 模型需要 TP+PP
# TP=4 在节点内（8 卡 / 节点，TP 组 = rank [0,1,2,3]）
# PP=2 跨节点（Stage 0 = 节点 0-1，Stage 1 = 节点 2-3）
# DP=4 自动计算
GBS=128  MBS=1
```

**场景 C：Qwen3-32B 长序列 256K，8 节点 × 16 卡 = 128 卡**

来自实际脚本 `tune_qwen3_32b_256K_full_pack_A3_ptd.sh`：

```bash
TP=8  PP=1  CP=8  DP=2
SEQ_LENGTH=262144
CP_TYPE='ulysses_cp_algo'
--recompute-granularity full \
--recompute-method block \
--recompute-num-layers 48    # 重计算节省超长序列的激活值显存
```

**场景 D：MOE 模型（如 Qwen3-MOE），4 节点 × 8 卡 = 32 卡**

```bash
TP=2  PP=2  EP=4  DP=2
# 假设 64 个专家，EP=4 → 每个 EP rank 持有 16 个专家
# DP=2 → 两份数据并行
```

### 10.5 权重转换注意事项

多机训练的 TP/PP 配置变化时，**必须重新转换权重**：

```bash
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \     # ← 与训练一致
    --target-pipeline-parallel-size 2 \   # ← 与训练一致
    --load-dir ./model_from_hf/qwen3_hf/ \
    --save-dir ./model_weights/qwen3_mcore_tp4pp2/ \
    --model-type-hf qwen3
```

### 10.6 检查点的自动转换（高级）

使用 `--enable-hf2mg-convert` 可以在训练启动时自动完成权重转换：

```bash
CKPT_ARGS="
    --enable-hf2mg-convert \
    --model-type-hf qwen3
"
# 直接传入 HF 权重路径，框架自动转换为 Megatron 格式并缓存
```

### 10.7 共享存储与非共享存储

```bash
# 所有节点挂载同一 NFS/共享文件系统
# （默认行为，数据和权重只需在共享路径存一份）

# 非共享存储（每个节点独立磁盘）
--no-shared-storage
# 数据和权重需在每个节点本地准备一份
```

### 10.8 多机快速启动脚本模板

```bash
#!/bin/bash
# ============ 多机训练启动模板 ============
# 在每个节点上运行此脚本，修改 NODE_RANK

export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=5400
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

# ---- 集群配置（所有节点相同） ----
NPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=192.168.1.100    # 主节点 IP
MASTER_PORT=6000

# ---- 当前节点配置（每个节点不同） ----
NODE_RANK=${1:-0}             # 从命令行参数接收：bash run.sh 0

# ---- 路径配置 ----
CKPT_LOAD_DIR="/shared/models/qwen3_mcore_tp4pp2"
CKPT_SAVE_DIR="/shared/checkpoints/qwen3_run1"
DATA_PATH="/shared/data/enwiki_text_document"
TOKENIZER_PATH="/shared/models/qwen3_hf"

# ---- 并行配置 ----
TP=4   PP=2   # DP = 32 / (4*2) = 4

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

torchrun \
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    ...  # 其余参数同单机脚本
```

在各节点分别执行：

```bash
# 节点 0
bash run.sh 0

# 节点 1（SSH 或作业调度器）
ssh node1 "cd /shared/MindSpeed-LLM && bash run.sh 1"

# 节点 2
ssh node2 "cd /shared/MindSpeed-LLM && bash run.sh 2"

# 节点 3
ssh node3 "cd /shared/MindSpeed-LLM && bash run.sh 3"
```

或使用 SLURM / pdsh 等集群工具批量启动。

---

## 11. 关键代码文件索引

| 文件 | 职责 |
|------|------|
| `core/parallel_state.py` | 进程组初始化（TP/PP/DP/EP/CP 通信组创建） |
| `core/tensor_parallel/layers.py` | TP 层实现（VocabParallelEmbedding 等） |
| `core/transformer/attention.py` | 2D TP 注意力实现 |
| `core/transformer/transformer_block.py` | PP 层分配（`get_num_layers_to_build`） |
| `core/pipeline_parallel/schedules.py` | PP 调度策略（1F1B/交错/RiPipe/DualPipe） |
| `core/context_parallel/` | CP 实现（Ring/Ulysses/Hybrid） |
| `core/transformer/moe/router.py` | EP Router（Group-Limited Greedy TopK） |
| `core/distributed/param_and_grad_buffer.py` | DP 梯度同步与通信重叠 |
| `fsdp2/distributed/mindspeed_parallel_engine.py` | FSDP2 并行引擎 |
| `fsdp2/distributed/parallel_engine_config.py` | FSDP2 并行配置声明 |
| `tasks/models/spec/qwen3_spec.py` | Qwen3 模型层规格（定义 TP 切分点） |
