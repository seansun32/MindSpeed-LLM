# 昇腾 NPU 适配实现分析：从 Megatron-LM 到 Ascend

## 1. 适配架构总览

MindSpeed-LLM 通过**三层架构**将 NVIDIA GPU 专属的 Megatron-LM 框架适配到华为昇腾 NPU 上，上游 Megatron-LM 代码**零修改**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     入口脚本 (pretrain_gpt.py)                      │
│  from mindspeed_llm import megatron_adaptor  ← 触发全部适配逻辑     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              MindSpeed-LLM（本项目）                      │       │
│  │  features_manager/ ── 70+ Feature 插件                   │       │
│  │  core/             ── NPU 优化算子 & 并行扩展             │       │
│  │  tasks/            ── 模型规格 & 任务适配                 │       │
│  └────────────────────────┬─────────────────────────────────┘       │
│                           │ 调用                                    │
│  ┌────────────────────────▼─────────────────────────────────┐       │
│  │              MindSpeed（中间件包）                         │       │
│  │  features_manager  ── 补丁管理框架                        │       │
│  │  core/             ── 2D TP / CP / 通信原语               │       │
│  │  te/               ── TransformerEngine NPU 适配          │       │
│  │  fsdp/             ── FSDP2 NPU 适配                     │       │
│  └────────────────────────┬─────────────────────────────────┘       │
│                           │ 调用                                    │
│  ┌────────────────────────▼─────────────────────────────────┐       │
│  │              torch_npu + CANN                             │       │
│  │  torch_npu         ── PyTorch NPU 后端                    │       │
│  │  HCCL              ── 集合通信库（替代 NCCL）             │       │
│  │  ACL               ── 昇腾计算库（替代 cuDNN/cuBLAS）     │       │
│  └────────────────────────┬─────────────────────────────────┘       │
│                           │                                         │
│  ┌────────────────────────▼─────────────────────────────────┐       │
│  │              Ascend NPU 硬件                              │       │
│  │  Atlas 800T A2 / Atlas 900                                │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              Megatron-LM（上游，不修改）                   │       │
│  │  megatron.core     ── Transformer 核心                    │       │
│  │  megatron.training ── 训练循环                            │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

**核心设计理念**：不 fork Megatron-LM，而是通过运行时 Monkey Patching 将 Megatron 的 CUDA 实现替换为 NPU 实现，做到上游可独立升级。

---

## 2. 适配触发链路

### 2.1 完整启动流程

```
pretrain_gpt.py
  │ Line 10: from mindspeed_llm import megatron_adaptor
  │
  ▼
mindspeed_llm/__init__.py
  │ backend = os.environ.get("TRAINING_BACKEND", "mcore")
  │ from mindspeed_llm.tasks import megatron_adaptor_v2 as megatron_adaptor
  │
  ▼
megatron_adaptor_v2.py（模块级代码，import 时立即执行）
  │
  ├─ Line 10: from torch_npu.contrib import transfer_to_npu   ← [A] 设备转换钩子
  │
  ├─ Line 11: from mindspeed.features_manager ... import MindSpeedFeaturesManager
  │
  └─ Line 92: FeatureAdaptor.execute()                        ← [B] 执行全部适配
        │
        ├── [1] args = get_mindspeed_llm_args()    ← 预解析命令行参数
        │         └── process_args_v2(parser)       ← 注册 MindSpeed-LLM 自定义参数
        │
        ├── [2] delete_lock_file()                  ← 清理 JIT 编译锁
        │
        ├── [3] MindSpeedFeaturesManager.apply_features_pre_patches(args)
        │         └── 遍历所有 Feature，执行 pre_register_patches()
        │              └── 创建 dummy 类、修复 import 依赖
        │
        ├── [4] MindSpeedFeaturesManager.apply_features_patches(args)
        │         └── 遍历所有 Feature，执行 register_patches()
        │              └── 将 Megatron 函数/类替换为 NPU 优化版本
        │
        └── [5] del sys.modules["transformer_engine"]  ← 清理 TE 模块引用
```

### 2.2 关键代码

```python
# megatron_adaptor_v2.py

from torch_npu.contrib import transfer_to_npu    # ← [A]
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

class FeatureAdaptor:
    @classmethod
    def execute(cls):
        args = FeatureAdaptor.get_mindspeed_llm_args()
        FeatureAdaptor.delete_lock_file()
        MindSpeedFeaturesManager.apply_features_pre_patches(args)  # ← [B] Phase 1
        MindSpeedFeaturesManager.apply_features_patches(args)      # ← [B] Phase 2
        if 'transformer_engine' in sys.modules:
            del sys.modules["transformer_engine"]

FeatureAdaptor.execute()  # ← import 时立即执行
```

---

## 3. `transfer_to_npu`：CUDA → NPU 全局设备映射

`torch_npu.contrib.transfer_to_npu` 是整个适配的**最底层基石**。它在 import 时通过 Monkey Patching 将 PyTorch 中所有 CUDA 相关的 API 映射到 NPU：

```
torch.cuda.is_available()        → torch.npu.is_available()
torch.cuda.device_count()        → torch.npu.device_count()
torch.cuda.set_device(i)         → torch.npu.set_device(i)
torch.cuda.current_device()      → torch.npu.current_device()
tensor.cuda()                    → tensor.npu()
model.cuda()                     → model.npu()
torch.cuda.synchronize()         → torch.npu.synchronize()
torch.cuda.memory_allocated()    → torch.npu.memory_allocated()
...
```

**效果**：Megatron-LM 中所有 `torch.cuda.xxx` 调用自动转发到 `torch.npu.xxx`，无需修改 Megatron 源码。

```python
# Megatron 原始代码（不需要改）：
model = model.cuda()              # 实际执行 model.npu()
torch.cuda.set_device(local_rank) # 实际执行 torch.npu.set_device(local_rank)
```

---

## 4. Feature Patching 机制：两阶段补丁注入

### 4.1 Feature 基类接口

每个 Feature 继承自 `MindSpeedFeature`，可实现以下生命周期方法：

```python
class MindSpeedFeature:
    # -------- 参数阶段 --------
    def register_args(self, parser):
        """注册自定义命令行参数"""

    # -------- Pre-Patch 阶段（Phase 1）--------
    def pre_register_patches(self, patch_manager, args):
        """创建 dummy 类、修复 import 依赖"""

    # -------- Patch 阶段（Phase 2）--------
    def register_patches(self, patch_manager, args):
        """核心：将 Megatron 函数/类替换为 NPU 优化版本"""

    # -------- 参数验证阶段 --------
    def pre_validate_args(self, args):
        """参数预验证和默认值设置"""
    def validate_args(self, args):
        """参数验证"""
    def post_validate_args(self, args):
        """参数后验证"""
```

### 4.2 补丁注册 API

```python
# 替换一个函数
patch_manager.register_patch(
    'megatron.core.transformer.dot_product_attention.DotProductAttention',
    CustomDotProductAttention     # 替换为 NPU FlashAttention
)

# 替换一个方法
patch_manager.register_patch(
    'megatron.core.transformer.attention.Attention.__init__',
    attention_init                # 替换初始化逻辑
)

# 创建 dummy 类（解决 import 依赖）
patch_manager.register_patch(
    'transformer_engine.pytorch.tensor.QuantizedTensor',
    torch.nn.Module,
    create_dummy=True             # 创建空壳类防止 ImportError
)
```

### 4.3 补丁执行时序图

```
时间线 →

[import megatron_adaptor_v2]
  │
  ├── Phase 1: apply_features_pre_patches()
  │     ├── TransformerEngineBasicFeature.pre_register_patches()
  │     │     └── 创建 QuantizedTensor dummy → 解决 TE import
  │     ├── RequirementsBasicFeature.pre_register_patches()
  │     │     └── 修复 apex 等 CUDA 库的 import 检查
  │     └── ...
  │
  ├── Phase 2: apply_features_patches()
  │     ├── MegatronBasicFeature.register_patches()
  │     │     ├── Norm → PTNorm（NPU 优化归一化）
  │     │     ├── TransformerLayer → NPU TransformerLayer
  │     │     └── 梯度同步 → NPU 优化同步
  │     │
  │     ├── TrainingBasicFeature.register_patches()
  │     │     ├── train() → NPU 训练循环
  │     │     ├── load_checkpoint → NPU 检查点加载
  │     │     └── build_pretraining_data_loader → NPU 数据加载
  │     │
  │     ├── FusionAttentionFeature.register_patches()
  │     │     └── DotProductAttention → CustomDotProductAttention
  │     │                               (torch_npu.npu_fusion_attention)
  │     │
  │     ├── CoCFeature.register_patches()
  │     │     └── 注入 CoC 计算通信融合
  │     │
  │     └── ... (70+ Features)
  │
  ▼
[import megatron.training]  ← 此时 megatron 代码已被全部补丁替换
[执行 pretrain()]          ← 使用的全是 NPU 优化版本
```

---

## 5. 70+ Feature 分类与核心适配点

### 5.1 Feature 注册清单

```python
# features_manager/__init__.py::create_features_list()

features_list = []
add_megatron_basic_features(features_list)    # Megatron 基础适配
add_context_parallel_features(features_list)  # 上下文并行
add_llm_features(features_list)               # LLM 任务特性
add_affinity_features(features_list)          # CPU 亲和性
add_fusions_features(features_list)           # 算子融合
add_recompute_features(features_list)         # 重计算
add_functional_features(features_list)        # 功能性（Profiler/确定性计算）
add_tensor_parallel_features(features_list)   # 张量并行
add_pipeline_parallel_features(features_list) # 流水线并行
add_transformer_features(features_list)       # Transformer 层
add_tokenizer_features(features_list)         # Tokenizer
add_distributed_features(features_list)       # 分布式
add_reuse_param_features(features_list)       # FP32 参数复用
add_swap_manage_features(features_list)       # 显存 Swap
add_moe_features(features_list)               # MOE 特性
add_hccl_buffer_features(features_list)       # HCCL 缓冲区
add_optimizer_features(features_list)         # 优化器
add_swap_optimizer_feature(features_list)     # 优化器 Swap
add_disable_gloo_group_feature(features_list) # 禁用 Gloo
add_high_availability_feature(features_list)  # 高可用
add_finetune_feature(features_list)           # 微调
add_ai_framework_feature(features_list)       # AI 框架
```

### 5.2 按适配层级分类

```
┌─────────────────────────────────────────────────────────────┐
│ 第 1 层：设备与通信基础                                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ transfer_to_npu    torch.cuda → torch.npu 全局映射      │ │
│ │ DisableGlooGroup   禁用 Gloo 后端（NPU 不支持）         │ │
│ │ HCCL Options       替换 NCCL 通信选项为 HCCL            │ │
│ │ HcclBuffer         HCCL 通信缓冲区自适应/固定大小       │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ 第 2 层：Megatron 核心替换                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ MegatronBasic      Norm/Layer/梯度同步 → NPU 版本       │ │
│ │ TrainingBasic       训练循环/检查点/数据加载 → NPU 版本  │ │
│ │ TransformerEngine   TE 模块 → MindSpeed TE（NPU）       │ │
│ │ ModelBasic          模型构建/参数初始化 → NPU 适配       │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ 第 3 层：NPU 算子融合                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ FusionAttention     FlashAttn → torch_npu.npu_fusion_*  │ │
│ │ FusedSwiglu         SwiGLU → NPU 融合算子               │ │
│ │ FusedSoftmax        Softmax → NPU 融合算子              │ │
│ │ FusedRotaryPosEmb   RoPE → NPU 融合旋转位置编码         │ │
│ │ FusedRMSNorm        RMSNorm → NPU 融合归一化            │ │
│ │ GroupedMatmul       GMM → NPU 分组矩阵乘法              │ │
│ │ FusedMoEPermute     MOE Permute → NPU 融合算子          │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ 第 4 层：并行策略优化                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ CoC                 计算通信融合（Communication over     │ │
│ │                     Computation，昇腾特有优化）          │ │
│ │ MC2                 MOE 计算通信融合                     │ │
│ │ TP2d                2D 张量并行通信优化                  │ │
│ │ OptimizeP2PComm     流水线 P2P 通信优化                 │ │
│ │ OptimizeSendRecv    Send/Recv 通信优化                  │ │
│ │ RiPipe              Recompute-in-Pipeline 调度          │ │
│ │ DualPipeV           双向流水线调度                       │ │
│ │ MoEAlltoAllOverlap  MOE AlltoAll 与计算重叠             │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ 第 5 层：显存与性能优化                                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ReuseFP32Param      FP32 参数复用节省显存               │ │
│ │ SwapAttention       注意力激活值 Swap 到 CPU             │ │
│ │ SmartSwap           智能显存 Swap 策略                  │ │
│ │ SwapOptimizer       优化器状态 Swap 到 CPU              │ │
│ │ RecomputeActivation 激活值重计算                        │ │
│ │ RecomputeNorm       Norm 层重计算                       │ │
│ │ ChunkLoss           分块 Loss 计算（减少激活值显存）     │ │
│ │ NPUDeterministic    NPU 确定性计算模式                  │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 关键适配实现详解

### 6.1 HCCL 替代 NCCL

Megatron-LM 使用 NCCL（NVIDIA Collective Communication Library）做分布式通信。MindSpeed-LLM 将其替换为 HCCL（Huawei Collective Communication Library）。

**关键代码**：`core/parallel_state.py:346`

```python
def get_nccl_options_wrapper(get_nccl_options):
    """将 Megatron 的 NCCL options 替换为 HCCL options"""
    @wraps(get_nccl_options)
    def wrapper(pg_name, nccl_comm_cfgs):
        if hasattr(torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options, "hccl_config"):
            try:
                # 创建 HCCL 专属 options
                options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                options.hccl_config = {"group_name": str(pg_name)}
                return options
            except Exception:
                return get_nccl_options(pg_name, nccl_comm_cfgs)  # 降级回 NCCL
        return get_nccl_options(pg_name, nccl_comm_cfgs)
    return wrapper
```

**进程组创建时使用**：

```python
# parallel_state.py: 所有 new_group 调用都经过这个 wrapper
group = torch.distributed.new_group(
    ranks, timeout=timeout,
    pg_options=megatron.core.parallel_state.get_nccl_options('tp_exp', nccl_comm_cfgs)
    #                                       ↑ 已被替换为 HCCL options
)
```

**FSDP2 路径直接使用 HCCL**：

```python
# train_fsdp2.py:184
torch.distributed.init_process_group(
    backend="hccl",   # ← 直接指定 HCCL
    rank=rank,
    world_size=world_size
)
```

### 6.2 PTNorm：归一化层 NPU 适配

Megatron 原生使用 NVIDIA TransformerEngine (TE) 的归一化层，MindSpeed-LLM 用 `PTNorm` 替换。

**关键代码**：`core/transformer/custom_layers/transformer_engine.py`

```python
class PTNorm:
    """将 Megatron TE 的 Norm 层替换为 NPU 兼容版本"""
    def __new__(cls, config, hidden_size, eps=1e-5):
        args = get_args()
        if config.normalization == "RMSNorm":
            if args.tp_2d:
                # 2D TP 下使用专用 RMSNorm（从 mindspeed 包导入）
                return RMSNorm2D(hidden_size, eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm())
            else:
                return RMSNorm(dim=hidden_size, eps=eps,
                    sequence_parallel=config.sequence_parallel)
        elif config.normalization == "LayerNorm":
            if args.tp_2d:
                return LayerNorm2D(hidden_size, eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm())
            else:
                return nn.LayerNorm(normalized_shape=hidden_size, eps=eps)
```

**MegatronBasicFeature 注册补丁**：

```python
def register_mcore_basic_patches(self, pm, args):
    from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm

    # 替换所有 Megatron 使用 Norm 的位置
    pm.register_patch('megatron.core.models.gpt.gpt_layer_specs.LNImpl', PTNorm)
    pm.register_patch('megatron.core.transformer.torch_norm.WrappedTorchNorm', PTNorm)
    pm.register_patch('megatron.core.transformer.transformer_block.LayerNormImpl', PTNorm)
    pm.register_patch('megatron.core.extensions.transformer_engine.TENorm', PTNorm)
```

### 6.3 FlashAttention NPU 适配

这是最核心的算子适配。Megatron 使用 CUDA FlashAttention，MindSpeed-LLM 替换为 NPU 专用 FlashAttention 族算子。

**关键代码**：`core/transformer/custom_dot_product_attention.py`

```python
class CustomDotProductAttention(DotProductAttention):
    """NPU FlashAttention 实现，覆盖 5 种执行路径"""

    def forward(self, query, key, value, attention_mask, ...):
        # -------- 路径选择 --------

        # 路径 1: 增量解码（推理，单 token）
        if use_kv_cache and query.shape[1] == 1:
            output = torch_npu.npu_incre_flash_attention(
                query, key, value,
                num_heads=n_head,
                input_layout="BSH",
                scale_value=self.scale
            )

        # 路径 2: Prompt 解码（推理，多 token + KV Cache）
        elif use_kv_cache:
            output = torch_npu.npu_prompt_flash_attention(
                query, key, value,
                num_heads=n_head,
                input_layout="BSH",
                sparse_mode=args.sparse_mode,
                scale_value=self.scale
            )

        # 路径 3: 稀疏注意力（训练，MLA 等场景）
        elif args.use_sparse_flash_attn:
            output = torch_npu.npu_sparse_flash_attention(
                query, key, value,
                sparse_indices=topk_indices,
                scale_value=self.scale,
                return_softmax_lse=True
            )

        # 路径 4: FA v2（训练，分离 RoPE）
        elif args.mla_fa_divide_qk:
            output = torch_npu.npu_fusion_attention_v2(
                query, key, value, n_head, args.shape_order,
                query_rope=query_rope,
                key_rope=key_rope,
                scale=self.scale
            )[0]

        # 路径 5: 标准 FlashAttention（训练，默认路径）
        else:
            output = torch_npu.npu_fusion_attention(
                query, key, value, n_head, args.shape_order,
                pse=pse,
                atten_mask=self.attention_mask,
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len,
                scale=self.scale,
                keep_prob=1 - self.attention_dropout.p,
                sparse_mode=args.sparse_mode
            )
```

**路径选择流程图**：

```
                          ┌─────────────────┐
                          │ FlashAttention   │
                          │ 入口             │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │ 使用 KV Cache？  │
                          └───┬─────────┬───┘
                           Yes│         │No
                    ┌─────────▼──┐  ┌───▼────────────┐
                    │ seq_len=1? │  │ sparse_attn?    │
                    └──┬──────┬──┘  └──┬──────────┬──┘
                    Yes│      │No   Yes│          │No
               ┌───────▼──┐  │  ┌─────▼────┐ ┌───▼─────────┐
               │ npu_incre │  │  │ npu_     │ │ divide_qk?  │
               │ _flash_   │  │  │ sparse_  │ └──┬──────┬───┘
               │ attention │  │  │ flash_   │ Yes│      │No
               └──────────┘  │  │ attention│    │  ┌───▼───────┐
                     ┌───────▼──┐└─────────┘    │  │ npu_      │
                     │ npu_     │         ┌─────▼──┤ fusion_   │
                     │ prompt_  │         │ npu_   │ attention │
                     │ flash_   │         │ fusion_│（默认路径）│
                     │ attention│         │ attn_v2│           │
                     └─────────┘         └────────┘└──────────┘
```

**支持的输入格式**：

| 格式 | Layout | 说明 |
|------|--------|------|
| `SBH` | [seq, batch, hidden] | Megatron 默认（sequence first） |
| `BSH` | [batch, seq, hidden] | 推理模式 |
| `BNSD` | [batch, heads, seq, dim] | 标准注意力格式 |
| `TND` | [total_tokens, heads, dim] | 变长序列打包格式 |

**FusionAttentionFeature 注册补丁**：

```python
def register_patches(self, patch_manager, args):
    from mindspeed_llm.core.transformer.custom_dot_product_attention import (
        CustomDotProductAttention
    )
    # 替换 Megatron 的标准注意力为 NPU FlashAttention
    patch_manager.register_patch(
        'megatron.core.transformer.dot_product_attention.DotProductAttention',
        CustomDotProductAttention
    )
```

### 6.4 TransformerEngine NPU 适配

NVIDIA TransformerEngine（TE）提供 FP8 和融合算子，MindSpeed-LLM 用 NPU 版本替换。

**关键代码**：`features_manager/megatron_basic/transformer_engine_basic.py`

```python
class TransformerEngineBasicFeature(MindSpeedFeature):
    def pre_register_patches(self, pm, args):
        # Phase 1: 创建 dummy 类，防止 Megatron import TE 时报错
        pm.register_patch(
            'transformer_engine.pytorch.tensor.QuantizedTensor',
            torch.nn.Module, create_dummy=True
        )

    def register_patches(self, pm, args):
        # Phase 2: 用 MindSpeed TE 替换 Megatron TE
        from mindspeed.te.pytorch.module.grouped_linear import (
            MindSpeedTEGroupedLinear,
            MindSpeedTEColumnParallelGroupedLinear,
            MindSpeedTERowParallelGroupedLinear
        )
        pm.register_patch(
            'megatron.core.extensions.transformer_engine.TEGroupedLinear',
            MindSpeedTEGroupedLinear
        )

        # FP8 支持
        if getattr(args, "fp8_format", False):
            from mindspeed.te.pytorch.module.linear import (
                TERowParallelLinear, TEColumnParallelLinear
            )
            pm.register_patch(
                'megatron.core.extensions.transformer_engine.TEColumnParallelLinear',
                TEColumnParallelLinear   # NPU FP8 列并行
            )
```

### 6.5 CoC：计算通信融合（昇腾特有）

CoC（Communication over Computation）是昇腾 NPU 的独特优化，将集合通信操作与矩阵计算在硬件级别融合执行。

```bash
# 训练脚本中启用
--use-ascend-coc       # 开启 CoC
--coc-fused-kernel     # 使用融合内核版本
```

**CoCFeature**（来自 `mindspeed` 包）将 TP 的 AllReduce/ReduceScatter 与 Linear 的 MatMul 在同一个 NPU kernel 中执行，消除通信等待时间。

```
传统方式：  MatMul → 等待 → AllReduce → 等待 → 下一层
CoC 方式：  MatMul + AllReduce（硬件融合执行，无等待）
```

---

## 7. `mindspeed` 中间件包提供的能力

`mindspeed` 是独立于 `mindspeed_llm` 的中间件包，提供昇腾 NPU 的通用优化能力：

```
mindspeed/
├── features_manager/           # Feature 补丁管理框架
│   ├── features_manager.py     #   MindSpeedFeaturesManager
│   └── feature.py              #   MindSpeedFeature 基类
│
├── core/                       # 核心并行与计算原语
│   ├── parallel_state.py       #   CP/2D TP 进程组初始化
│   ├── distributed/            #   参数/梯度缓冲区同步
│   ├── tensor_parallel/
│   │   └── tp_2d/              #   2D 张量并行
│   │       ├── group_api_2d.py #     TPX/TPY 通信接口
│   │       ├── layernorm_2d.py #     2D TP LayerNorm
│   │       ├── rms_norm_2d.py  #     2D TP RMSNorm
│   │       └── parallel_linear_2d.py  # 2D TP 线性层
│   ├── context_parallel/       #   上下文并行实现
│   └── pipeline_parallel/      #   RiPipe / DualPipe 调度
│
├── te/                         # TransformerEngine NPU 适配
│   └── pytorch/module/
│       ├── linear.py           #   TE Linear（NPU FP8）
│       └── grouped_linear.py   #   TE GroupedLinear
│
├── fsdp/                       # FSDP NPU 适配
│   ├── utils/
│   │   ├── device.py           #   set_accelerator_compatible(torch.npu)
│   │   ├── random.py           #   NPU 随机种子管理
│   │   └── torch_patch.py      #   apply_hccl_premul_sum_patch()
│   └── distributed/            #   FSDP 分片与并行引擎
│
└── deprecate.py                # AutoExecuteFunction 等工具
```

**关键接口**：

```python
# 设备兼容
from mindspeed.fsdp.utils.device import set_accelerator_compatible
set_accelerator_compatible(torch.npu)  # 将 NPU 标记为主加速器

# HCCL 优化
from mindspeed.fsdp.utils.torch_patch import apply_hccl_premul_sum_patch
apply_hccl_premul_sum_patch()  # AllReduce premul_sum 优化

# 2D 张量并行通信
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import (
    TPXCollectiveComm,  TPXOverlapCollectiveComm,
    TPYCollectiveComm,  TPYOverlapCollectiveComm,
)
```

---

## 8. NVIDIA CUDA 与 Ascend NPU 的对应关系

```
┌──────────────────────┬───────────────────────┬──────────────────────┐
│       NVIDIA 栈      │     Ascend NPU 栈      │       替换方式        │
├──────────────────────┼───────────────────────┼──────────────────────┤
│ torch.cuda           │ torch.npu              │ transfer_to_npu     │
│ CUDA Driver          │ Ascend Driver          │ 驱动层               │
│ NCCL                 │ HCCL                   │ get_nccl_options_    │
│                      │                        │ wrapper              │
│ cuDNN / cuBLAS       │ ACL (Ascend CL)        │ torch_npu 内部       │
│ FlashAttention       │ npu_fusion_attention   │ FusionAttention      │
│ (triton / cutlass)   │ (NPU FA 族算子)         │ Feature              │
│ TransformerEngine    │ MindSpeed TE           │ TransformerEngine    │
│ (FP8/融合算子)        │ (NPU FP8/融合算子)      │ BasicFeature         │
│ apex (LayerNorm 等)  │ PTNorm (NPU Norm)      │ MegatronBasic        │
│                      │                        │ Feature              │
│ CUDA Streams         │ NPU Streams            │ transfer_to_npu     │
│ cudaMalloc           │ npuMalloc              │ transfer_to_npu     │
│ nvprof / NSight      │ msprof / CANN Profiler │ ProfilingFeature     │
└──────────────────────┴───────────────────────┴──────────────────────┘
```

---

## 9. 一个完整的适配案例：Attention 层

以 Qwen3 模型的 Attention 层为例，追踪从 Megatron 原始代码到 NPU 执行的完整适配链路：

```
[1] Qwen3 Spec 定义（不变）
    tasks/models/spec/qwen3_spec.py:
    self_attention = ModuleSpec(
        module=SelfAttention,
        submodules=SelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear,     # Megatron TP 列并行
            core_attention=DotProductAttention,    # ← 将被替换
            linear_proj=RowParallelLinear,         # Megatron TP 行并行
        )
    )

[2] FusionAttentionFeature 替换 core_attention（import 时执行）
    DotProductAttention → CustomDotProductAttention

[3] MegatronBasicFeature 替换 Norm 层（import 时执行）
    TENorm / WrappedTorchNorm → PTNorm

[4] 如果启用 2D TP, attention init wrapper 替换 linear_qkv / linear_proj
    ColumnParallelLinear → ParallelLinear2D
    RowParallelLinear → ParallelLinear2D

[5] 训练时前向传播执行路径：
    input
      → PTNorm(input)                              # NPU RMSNorm
      → ColumnParallelLinear(input) → Q, K, V       # NPU MatMul + TP AllGather
      → PTNorm(Q), PTNorm(K)                        # QK LayerNorm（Qwen3 特有）
      → apply_rotary_pos_emb(Q, K)                  # NPU 融合 RoPE
      → CustomDotProductAttention.forward(Q, K, V)   # torch_npu.npu_fusion_attention
      → RowParallelLinear(output)                    # NPU MatMul + TP AllReduce
      → residual + output                            # NPU Add
```

---

## 10. 适配对性能的影响

### 10.1 相比原生 Megatron + CUDA 的差异

| 维度 | 影响 | 说明 |
|------|------|------|
| 设备映射 | 几乎无开销 | `transfer_to_npu` 仅是 API 转发 |
| 算子融合 | 性能提升 | NPU 融合算子（FA/SwiGLU/RMSNorm）通常与 CUDA 版本持平或更优 |
| 通信库 | 取决于网络 | HCCL 在昇腾硬件上通常优于 NCCL 模拟 |
| Feature Patching | 一次性开销 | 仅在 import 时执行，训练中无开销 |
| CoC | 显著提升 | 计算通信融合是昇腾独有优化，可隐藏通信延迟 |

### 10.2 NPU 独有的优化参数

```bash
# 昇腾 NPU 推荐的训练优化参数组合
--use-flash-attn              # NPU FlashAttention
--use-fused-rotary-pos-emb    # NPU 融合 RoPE
--use-fused-swiglu            # NPU 融合 SwiGLU
--use-fused-rmsnorm           # NPU 融合 RMSNorm
--use-ascend-coc              # CoC 计算通信融合
--coc-fused-kernel            # CoC 融合内核
--reuse-fp32-param            # FP32 参数复用
--overlap-grad-reduce         # 梯度通信重叠
--overlap-param-gather        # 参数通信重叠
```

---

## 11. 关键代码文件索引

| 文件 | 职责 |
|------|------|
| `tasks/megatron_adaptor_v2.py` | 适配入口，FeatureAdaptor.execute() |
| `features_manager/__init__.py` | 70+ Feature 注册清单 |
| `features_manager/megatron_basic/megatron_basic.py` | Norm/Layer/梯度同步替换 |
| `features_manager/megatron_basic/training_basic.py` | 训练循环/检查点替换 |
| `features_manager/megatron_basic/transformer_engine_basic.py` | TE → MindSpeed TE 替换 |
| `features_manager/transformer/flash_attention/fusion_attention_feature.py` | FlashAttention 替换 |
| `core/transformer/custom_dot_product_attention.py` | NPU FlashAttention 5 路径实现 |
| `core/transformer/custom_layers/transformer_engine.py` | PTNorm 实现 |
| `core/transformer/mlp.py` | MLP 2D TP 适配 |
| `core/transformer/attention.py` | Attention 2D TP 适配 |
| `core/parallel_state.py` | HCCL options + 进程组扩展 |
