# MindSpeed-LLM 核心代码分析：训练流程、数据流转与模型初始化

## 1. 训练入口总览

项目提供多个顶层入口脚本，各自对应不同的训练/推理场景：

| 入口脚本 | 场景 | 训练后端 |
|----------|------|----------|
| `pretrain_gpt.py` | GPT 预训练 | Megatron-core |
| `posttrain_gpt.py` | 微调（SFT/DPO） | Megatron-core |
| `pretrain_mamba.py` | Mamba 模型预训练 | Megatron-core |
| `rlhf_gpt.py` | RLHF 训练 | Megatron-core |
| `train_fsdp2.py` | 预训练/微调 | FSDP2（新方案） |
| `inference.py` | 推理 | — |
| `evaluation.py` | 评估 | — |

所有 Megatron 路径的入口在执行前，**首先触发特性注入**：

```python
# pretrain_gpt.py 第 10 行
from mindspeed_llm import megatron_adaptor  # ← 触发 FeatureAdaptor.execute()
```

---

## 2. 特性注入机制（启动前的关键步骤）

### 2.1 触发链路

```
import mindspeed_llm
  └── __init__.py: backend = os.environ.get("TRAINING_BACKEND", "mcore")
        └── if backend == "mcore":
              from mindspeed_llm.tasks import megatron_adaptor_v2 as megatron_adaptor
```

```
megatron_adaptor_v2.py（模块级代码，import 时立即执行）:
  └── FeatureAdaptor.execute()
        ├── args = FeatureAdaptor.get_mindspeed_llm_args()   # 预解析命令行参数
        ├── MindSpeedFeaturesManager.apply_features_pre_patches(args)  # 应用前置补丁
        └── MindSpeedFeaturesManager.apply_features_patches(args)      # 应用主补丁
```

### 2.2 特性列表注册

`features_manager/__init__.py` 在模块加载时自动注册全部特性：

```python
@AutoExecuteFunction
def set_default_features_list():
    MindSpeedFeaturesManager.set_features_list(create_features_list())
```

`create_features_list()` 按类别组织 50+ 特性：

```
Megatron 基础特性 → 上下文并行 → LLM 特性 → CPU 亲和性 → 算子融合
→ 重计算 → 功能性 → 张量并行 → 流水线并行 → Transformer
→ Tokenizer → 分布式 → FP32 参数复用 → Swap → MOE
→ HCCL 缓冲区 → 优化器 → 高可用 → 微调 → AI 框架
```

**设计意义**：所有优化特性通过 Monkey Patching 方式注入到 Megatron-LM 的核心代码中，使得**上游 Megatron 代码不需要修改**，MindSpeed-LLM 的定制逻辑以插件形式叠加在上面。

---

## 3. Megatron 路径：预训练完整流程

### 3.1 总体流程图

```
pretrain_gpt.py::main()
  │
  │  pretrain(
  │      train_valid_test_datasets_provider,  ← 数据集构建函数
  │      model_provider,                      ← 模型构建函数
  │      ModelType.encoder_or_decoder,        ← 模型类型
  │      forward_step                         ← 前向计算函数
  │  )
  │
  ▼
training/training.py::pretrain()
  │
  ├── [1] initialize_megatron()               ← 初始化分布式环境
  │     ├── parse_args()                      ← 解析参数（含 MindSpeed-LLM 自定义参数）
  │     ├── validate_args()                   ← 验证参数
  │     ├── set_global_variables()            ← 设置全局变量（args/tokenizer/tensorboard）
  │     ├── _initialize_distributed()         ← 初始化 PyTorch 分布式 + 模型并行组
  │     ├── _set_random_seed()                ← 设置随机种子
  │     ├── _compile_dependencies()           ← 编译数据集索引构建器
  │     └── [Hook] 数据预处理 / 权重转换      ← initialize_megatron_wrapper 注入
  │
  ├── [2] set_jit_fusion_options()            ← JIT 编译选项
  │
  ├── [3] build_train_args()                  ← 构建模型 + 优化器 + 数据
  │     ├── setup_model_and_optimizer()       ← 模型初始化 + 优化器创建
  │     │     ├── model_provider_func_wrapper()  ← 模型后处理（LoRA 注入等）
  │     │     ├── get_model()                 ← DDP 封装
  │     │     └── get_megatron_optimizer()    ← 创建分布式优化器
  │     └── build_train_valid_test_data_iterators()  ← 数据迭代器构建
  │
  ├── [4] train()                             ← 主训练循环
  │     └── while iteration < train_iters:
  │           ├── train_step()                ← 单步训练
  │           │     ├── forward_backward_func()  ← 前向 + 反向（含流水线调度）
  │           │     ├── optimizer.step()      ← 参数更新
  │           │     └── opt_param_scheduler.step()  ← 学习率更新
  │           ├── training_log()              ← 日志记录
  │           ├── evaluate_and_print_results()  ← 定期评估
  │           └── save_checkpoint()           ← 定期保存检查点
  │
  ├── [5] evaluate（验证集 / 测试集）
  └── [6] 清理与结束
```

### 3.2 关键函数签名

`pretrain()` 接收四个用户定义的回调函数，这是 Megatron 路径的核心设计模式：

```python
def pretrain(
    train_valid_test_dataset_provider,  # (train_val_test_num_samples) → (train_ds, valid_ds, test_ds)
    model_provider,                     # (pre_process, post_process) → model
    model_type,                         # ModelType 枚举
    forward_step_func,                  # (data_iterator, model) → (output_tensor, loss_func)
    ...
)
```

---

## 4. 模型初始化详解

### 4.1 model_provider — 模型构建

位于 `pretrain_gpt.py:42`，是预训练场景的模型工厂函数。

**执行流程**：

```
model_provider(pre_process, post_process)
  │
  ├── config = core_transformer_config_from_args(args)    ← 从 args 构建 TransformerConfig
  │
  ├── [mcore 路径] args.use_legacy_models == False（默认）
  │     ├── transformer_layer_spec = get_gpt_layer_local_spec()  ← 或从 args.spec 导入自定义 spec
  │     └── model = GPTModel(config, transformer_layer_spec, ...)
  │
  └── [legacy 路径] args.use_legacy_models == True
        └── model = megatron.legacy.model.GPTModel(config, ...)
```

### 4.2 Transformer Layer Spec — 模型规格系统

模型的层结构通过 `ModuleSpec` 声明式定义，不同模型通过不同 spec 文件来定制。

以 `tasks/models/spec/qwen3_spec.py` 为例：

```python
layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=PTNorm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=PTNorm,
                k_layernorm=PTNorm,
            ),
        ),
        pre_mlp_layernorm=PTNorm,
        mlp=_get_mlp_module_spec(...),
        ...
    ),
)
```

spec 通过 `--spec` 参数指定，Megatron 通过 `import_module(args.spec)` 动态加载。

### 4.3 model_provider_func_wrapper — 模型后处理

`training/training.py:101`，在模型构建后应用额外逻辑：

```
model_provider_func_wrapper(model_provider_func)
  │
  ├── model = model_provider_func(...)       ← 原始模型构建
  ├── [可选] 注入 Fused MLP forward          ← args.use_fused_mlp
  └── [可选] 注入 LoRA 适配器                 ← is_enable_lora()
        ├── lora_config = LoraConfig(r, alpha, target_modules, ...)
        ├── model = get_peft_model(model, lora_config)
        ├── [可选] activate_lu_lora_layers()  ← LU-LoRA
        └── 设置 sequence_parallel 属性
```

### 4.4 DDP 封装与优化器创建

在 `setup_model_and_optimizer()` (Megatron 提供)中：

```
model → 转移到 GPU → FP16/BF16 封装 → DDP 封装 → 返回 model_list
optimizer = get_megatron_optimizer(model_list)  → DistributedOptimizer 或 Float16OptimizerWithFloat16Params
opt_param_scheduler = OptimizerParamScheduler(optimizer, ...)
```

---

## 5. 数据流转详解

### 5.1 数据集构建

**预训练路径** (`pretrain_gpt.py:269`)：

```
train_valid_test_datasets_provider(train_val_test_num_samples)
  │
  ├── config = GPTDatasetConfig(seed, seq_length, blend, tokenizer, ...)
  └── BlendedMegatronDatasetBuilder(GPTDataset, num_samples, config).build()
        └── 返回 (train_ds, valid_ds, test_ds)
```

数据集基于 Megatron 的 `IndexedDataset`（mmap 内存映射），支持：
- 多数据源混合（Blended Dataset）
- 按比例切分 train/valid/test

**微调路径** (`tasks/posttrain/utils.py`)：

```
train_valid_test_datasets_provider(train_val_test_num_samples)
  └── 从 JSON/JSONL 文件加载指令数据
        ├── Alpaca 格式
        ├── ShareGPT 格式
        └── 自定义对话格式
```

### 5.2 数据迭代器构建

```
build_train_valid_test_data_iterators(datasets_provider)
  │
  ├── datasets = datasets_provider(num_samples)
  ├── DataLoader(dataset, sampler=MegatronPretrainingSampler, ...)
  └── 返回 (train_iter, valid_iter, test_iter)
```

对于**虚拟流水线并行**，每个虚拟阶段有独立的数据迭代器。

### 5.3 Batch 获取与分发

**预训练 get_batch** (`pretrain_gpt.py:108`)：

```
get_batch(data_iterator)
  │
  ├── batch = get_batch_on_this_tp_rank(data_iterator)
  │     └── TP rank 0 从迭代器取数据，广播到同组其他 rank
  │
  ├── [可选] generate_mtp_batch_list_on_this_tp_rank(batch)  ← 多 Token 预测
  │
  └── batch = get_batch_on_this_cp_rank(batch)
        └── 按序列维度切分到各 CP rank

  返回: (tokens, labels, loss_mask, attention_mask, position_ids)
```

**微调 SFTTrainer.get_batch** (`tasks/posttrain/sft/sft_trainer.py:34`)：

```
get_batch(data_iterator)
  │
  ├── data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
  │     └── 从 data_iterator 取数据并广播到 TP 组
  │
  ├── tokens = data_b['input_ids']
  ├── labels = data_b['labels']
  ├── loss_mask = where(labels == IGNORE_INDEX, 0, 1)  ← 忽略 padding 位置的 loss
  │
  └── batch = get_batch_on_this_cp_rank(batch)  ← CP 切分

  返回: (tokens, labels, loss_mask, attention_mask, position_ids)
```

### 5.4 数据在流水线中的流动

```
forward_step(data_iterator, model)
  │
  ├── (tokens, labels, loss_mask, attention_mask, position_ids) = get_batch(data_iterator)
  │     注意：中间 PP 阶段不需要原始数据，get_batch 返回 (None,) * 5
  │
  ├── output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
  │     ├── PP 第一阶段：嵌入层 → Transformer 块 → 激活值通过 P2P 发送给下一阶段
  │     ├── PP 中间阶段：接收激活值 → Transformer 块 → 发送给下一阶段
  │     └── PP 最后阶段：接收激活值 → Transformer 块 → LM Head → Loss
  │
  └── return output_tensor, partial(loss_func, loss_mask)
```

---

## 6. 训练循环详解

### 6.1 单步训练 train_step()

```
train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config)
  │
  ├── optimizer.zero_grad()
  │
  ├── forward_backward_func(...)    ← 根据 PP 配置选择调度策略
  │     ├── 无 PP: forward_backward_no_pipelining()
  │     │     └── 对每个 micro-batch:
  │     │           ├── output = forward_step_func(data_iter, model)
  │     │           └── backward(output)
  │     │
  │     ├── 交错式 PP: forward_backward_pipelining_with_interleaving()
  │     │     └── 1F1B 调度，虚拟阶段交错执行
  │     │
  │     └── 非交错 PP: forward_backward_pipelining_without_interleaving()
  │           └── 先 warmup(全 forward)，再 1F1B，最后 cooldown(全 backward)
  │
  ├── finalize_model_grads()        ← 梯度 AllReduce（跨 DP 组）
  │
  ├── optimizer.step()              ← 参数更新（含梯度裁剪、loss scaling）
  │
  └── opt_param_scheduler.step()    ← 学习率调度
```

### 6.2 Loss 计算

```
loss_func(loss_mask, output_tensor)
  │
  ├── losses = output_tensor.float()
  ├── loss = sum(losses.view(-1) * loss_mask.view(-1))  ← 按 loss_mask 加权
  │
  ├── [CP] all_reduce(loss, group=cp_group)    ← 上下文并行 reduce
  │
  ├── [校验] NaN / Inf / Spiky Loss 检测
  │
  ├── [DP] all_reduce(reporting_loss, group=dp_group)  ← 数据并行 reduce（用于日志）
  │
  └── return (loss, local_num_tokens, {'lm loss': (reporting_loss, total_tokens)})
```

---

## 7. 微调路径：PostTrain 流程

### 7.1 入口

```
posttrain_gpt.py::launch()
  │
  ├── from mindspeed_llm import megatron_adaptor  ← 特性注入
  └── AutoTrainer().train()
```

### 7.2 AutoTrainer 调度

```
tasks/posttrain/launcher.py::AutoTrainer
  │
  ├── initialize_megatron()                    ← 环境初始化
  ├── stage = args.stage                       ← "sft" / "dpo"
  └── trainer = get_trainer(stage)             ← 工厂方法
        ├── "sft" → SFTTrainer()
        └── "dpo" → DPOTrainer()
```

### 7.3 BaseTrainer 初始化

```
BaseTrainer.__init__()
  │
  ├── args = get_args()
  ├── initialize()
  │     ├── set_jit_fusion_options()
  │     ├── synchronize_start_time()
  │     └── build_train_args(...)              ← 复用预训练的构建逻辑
  │           ├── setup_model_and_optimizer(model_provider, ...)
  │           └── build_train_valid_test_data_iterators(...)
  │
  └── train()
        └── train(forward_step_func, model, optimizer, ...)  ← 复用同一个训练循环
```

**关键点**：SFTTrainer 和 DPOTrainer 继承 BaseTrainer，仅重写 `get_batch()`、`loss_func()`、`forward_step()` 三个方法，训练循环完全复用。

---

## 8. FSDP2 路径：新一代训练框架

### 8.1 入口与调度

```
train_fsdp2.py::AutoTrainer
  │
  ├── backend = os.environ.get("TRAINING_BACKEND", "mcore")
  │
  ├── "mcore" → McoreAutoTrainer()
  │     ├── initialize_megatron()
  │     └── FSDP2PretrainTrainer / FSDP2SFTTrainer
  │
  └── "mindspeed_fsdp" → MindSpeedAutoTrainer()
        ├── _parse_args()               ← HfArgumentParser 风格参数
        ├── _initialize()               ← 分布式环境 + NPU 设备 + 随机种子
        ├── _build_model()              ← ModelFactory.create()
        ├── _build_tokenizer()          ← TokenizerFactory.create()
        ├── _build_data_manager()       ← DataFactory.create()
        ├── _build_optimizer()          ← OptimizerFactory.create()
        ├── _build_scheduler()          ← SchedulerFactory.create()
        ├── _build_checkpointer()       ← CheckpointManager
        └── Trainer(model, optimizer, lr_scheduler, data_manager, ...)
```

### 8.2 FSDP2 模型初始化

```
ModelFactory.create(model_args, parallel_args)
  │
  ├── hf_config = AutoConfig.from_pretrained(model_name_or_path)
  │
  ├── [Meta Device 模式] init_model_with_meta_device == True
  │     └── with torch.device("meta"):
  │           model = AutoModelForCausalLM.from_config(hf_config)
  │
  ├── [CPU 模式] init_model_with_meta_device == False
  │     ├── train_from_scratch: AutoModelForCausalLM.from_config(hf_config)
  │     └── 否则: ModelLoader.load(model_name_or_path)  ← 加载预训练权重
  │
  └── MindSpeedParallelEngine.parallelize(model, config)
        ├── TP（张量并行）封装
        ├── EP（专家并行）封装
        ├── CP（上下文并行）封装
        └── FSDP 封装
```

### 8.3 FSDP2 数据流

```
DataFactory.create(data_manager_type, ...)
  │
  ├── "lf" → LFDataManager  (LlamaFactory 风格，用于 SFT)
  │     └── create_train_dataloader()
  │           ├── dataset = get_dataset(data_args, tokenizer, template)
  │           ├── collator = SFTDataCollatorWith4DAttentionMask(tokenizer)
  │           └── DataLoader(dataset, sampler, collator, ...)
  │
  └── "megatron" → MegatronDataManager  (Megatron 风格，用于预训练)
        └── create_train_dataloader()
              ├── train_ds = train_valid_test_datasets_provider(...)
              ├── sampler = MegatronPretrainingSampler(...)
              └── DataLoader(train_ds, sampler, ...)
```

### 8.4 FSDP2 训练循环

```
Trainer.train()
  │
  ├── for epoch in range(num_train_epochs):
  │     for step, batch in enumerate(train_dataloader):
  │       │
  │       ├── batch → GPU (input_ids, attention_mask, labels, ...)
  │       │
  │       ├── loss = model(**batch).loss / gradient_accumulation_steps
  │       │
  │       ├── loss.backward()
  │       │
  │       ├── [每 gradient_accumulation_steps 步]
  │       │     ├── clip_grad_norm(model.parameters(), max_grad_norm)
  │       │     ├── optimizer.step()
  │       │     ├── lr_scheduler.step()
  │       │     └── optimizer.zero_grad()
  │       │
  │       ├── 日志记录（loss、lr、throughput）
  │       └── 定期 checkpoint 保存
  │
  └── 保存最终模型
```

**与 Megatron 路径的核心区别**：
- FSDP2 使用 HuggingFace 模型 + PyTorch 原生 FSDP，不依赖 Megatron 的并行原语
- 训练循环更简洁，类似 HuggingFace Trainer
- 数据处理直接使用 HuggingFace datasets 库

---

## 9. 参数系统

### 9.1 Megatron 路径

参数解析通过装饰器链扩展 Megatron 原生参数：

```
training/arguments.py::parse_args_decorator(parse_args)
  └── extra_args_provider_decorator(extra_args_provider)
        └── process_args_v2(parser)
              ├── _add_fusion_op_args()      ← 算子融合参数
              ├── _add_network_size_args()    ← 网络结构参数
              ├── _add_lora_args()            ← LoRA 参数
              ├── _add_data_args()            ← 数据参数
              ├── _add_training_args()        ← 训练参数
              └── ...                         ← 更多参数组
```

`arguments.py` 是项目最大的单文件之一（77KB），定义了 MindSpeed-LLM 所有自定义命令行参数。

### 9.2 FSDP2 路径

使用 dataclass 风格的参数定义：

```python
@dataclass
class Arguments:
    model: ModelArguments          # 模型路径、类型等
    data: DataArguments            # 数据路径、格式等
    parallel: ParallelArguments    # TP/PP/CP/EP 大小等
    training: TrainingArguments    # LR、batch_size、epochs 等
```

通过 `fsdp2_parse_args()` 从 YAML 配置文件解析。

---

## 10. 两条路径的对比

| 维度 | Megatron 路径 | FSDP2 路径 |
|------|---------------|------------|
| **模型来源** | Megatron GPTModel + 自定义 Spec | HuggingFace AutoModelForCausalLM |
| **并行封装** | Megatron 原生 TP/PP/DP | PyTorch FSDP + MindSpeed 并行引擎 |
| **数据格式** | IndexedDataset（mmap 二进制） | HuggingFace datasets / Megatron |
| **训练循环** | 自定义 train() + train_step() | 类 HuggingFace Trainer 风格 |
| **参数系统** | argparse + 装饰器扩展 | dataclass + YAML |
| **特性注入** | features_manager Monkey Patching | 模型注册表 + 并行引擎配置 |
| **成熟度** | 成熟，100+ 模型支持 | 较新，逐步扩展中 |
| **环境变量** | `TRAINING_BACKEND=mcore`（默认） | `TRAINING_BACKEND=mindspeed_fsdp` |

---

## 11. 完整数据流转示意图

### Megatron 预训练路径

```
原始文本文件
  │ preprocess_data.py（离线预处理）
  ▼
IndexedDataset (.bin + .idx)
  │ BlendedMegatronDatasetBuilder
  ▼
GPTDataset (mmap 加载)
  │ MegatronPretrainingSampler
  ▼
DataLoader
  │ get_batch_on_this_tp_rank()  ← TP rank 0 取数据并广播
  ▼
{tokens, labels, loss_mask, attention_mask, position_ids}
  │ get_batch_on_this_cp_rank()  ← 按序列维度切分到 CP 组
  ▼
切分后的 batch
  │ model(tokens, position_ids, attention_mask, labels)
  ▼
output_tensor (loss per token)
  │ loss_func(loss_mask, output_tensor)
  │   ├── loss_mask 加权求和
  │   ├── CP all_reduce
  │   └── DP all_reduce (reporting only)
  ▼
scalar loss → backward → optimizer.step()
```

### FSDP2 SFT 路径

```
JSON/JSONL 指令数据
  │ get_dataset() (HuggingFace datasets)
  ▼
Dataset (tokenized)
  │ SFTDataCollatorWith4DAttentionMask
  ▼
DataLoader
  │ batch → GPU
  ▼
{input_ids, attention_mask, labels}
  │ model(**batch)
  ▼
CausalLMOutput.loss
  │ loss.backward()
  ▼
optimizer.step() → lr_scheduler.step()
```

---

## 12. 关键代码文件索引

| 文件 | 行数 | 核心职责 |
|------|------|----------|
| `mindspeed_llm/__init__.py` | 20 | 后端选择，触发特性注入 |
| `tasks/megatron_adaptor_v2.py` | 92 | FeatureAdaptor 执行入口 |
| `features_manager/__init__.py` | 344 | 特性列表注册 |
| `training/training.py` | ~850 | pretrain() + train() 主循环 |
| `training/initialize.py` | 277 | Megatron 初始化 + 权重转换钩子 |
| `training/arguments.py` | ~2000 | 全量参数定义 |
| `tasks/posttrain/launcher.py` | 53 | 微调 AutoTrainer 调度 |
| `tasks/posttrain/base/base_trainer.py` | ~200 | 微调基类（model_provider + train） |
| `tasks/posttrain/sft/sft_trainer.py` | ~200 | SFT get_batch / loss_func / forward_step |
| `train_fsdp2.py` | 315 | FSDP2 AutoTrainer 入口 |
| `fsdp2/train/mindspeed_trainer.py` | ~700 | FSDP2 Trainer 训练循环 |
| `fsdp2/models/model_factory.py` | ~300 | FSDP2 模型工厂 |
| `fsdp2/data/data_factory.py` | ~200 | FSDP2 数据工厂 |
