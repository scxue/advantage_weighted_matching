# SD3 任务：epoch 边界 resume 与逻辑 32-rank 输入对齐

## 使用

`geneval_sd3_no_cfg_1node`、`geneval_sd3_no_cfg_2nodes`、`geneval_sd3_no_cfg_4nodes`
默认启用 `logical_ranks=32`，`checkpoint_every_steps=10`。
三个入口共享同一套配置：beta=0.003、ghuber=0.5、with-init、
base-KL CFG=4.5、shared prompt training noise；仅节点数对应的执行参数、
运行名称和保存路径不同。FP16 overflow 的 step/scheduler 行为未调整。
不再提供单独的 wo-init 和 kl_beta_0003 配置入口。
1-node 使用 8 GPU、每卡 4 个逻辑 rank，rollout batches=36、梯度累积=72。

OCR/PickScore 的现有 1-node、4-node 入口也启用完整 resume/逻辑 32-rank，
并统一 ghuber=0.5、with-init、base-KL CFG=4.5、shared training noise。
保留任务自身的 KL 系数、rollout 参数、batch size 和评估频率：

| 任务 | KL beta | logical_rounds | 每逻辑 rank 图片数 | 训练 batch | 每逻辑 rank microbatches | 全局图片/epoch |
| --- | --- | --- | --- | --- | --- | --- |
| Geneval / OCR | 0.003 | 9 | 54 | 3 | 18 | 1728 |
| PickScore | 0.0001 | 3 | 18 | 2 | 9 | 576 |

`logical_rounds` 是每逻辑 rank 的 rollout batch 数。物理进程负责几个逻辑
rank，就把 rollout batches 和梯度累积相应乘几倍；不混合不同逻辑 rank
的 shuffle。只支持每节点 8 GPU 的对应入口；启动会检查配置与进程数。
OCR/PickScore 使用 train.txt/test.txt，Geneval 使用 metadata JSONL，
两者都将实际数据文件内容纳入恢复签名。

旧 OCR/PickScore 1-node 的输入顺序会改成对应任务旧 4-node 顺序，
不是保留旧 1-node 轨迹。四卡/八卡等更小进程数仅供显式调整配置的测试。

在原来的启动命令中追加：

```bash
--config.resume_from=/absolute/path/to/save_dir
```

也可以指定 `save_dir/checkpoints/checkpoint-100`。指定 run 目录时选择编号最大的
完整 checkpoint；`.incomplete-*` 和没有完成标记的目录不会用于恢复。
恢复沿用 checkpoint 中的 `save_dir`、run name 和 W&B run ID。
请只加载自己信任的 checkpoint：完整状态使用 PyTorch pickle 格式。

例如，从表现较好的 beta=0.003 的 4-node 实验改为 2-node 继续训练，使用原本
2-node 的 Slurm/Accelerate 启动方式，并传入：

```bash
--config=config/dgx_awm.py:geneval_sd3_no_cfg_2nodes \
--config.resume_from=/absolute/path/to/the/4node/save_dir
```

训练参数必须与保存时相同；三个入口现在都默认 beta=0.003，无需另行覆盖。
恢复会检查训练配置、数据 SHA256、核心代码和依赖版本；
节点数对应的 rollout/accumulation 参数允许变化，训练总轮数、日志目录和保存频率
也允许变化。签名不一致会明确报错，不会悄悄进行 weight-only warm start。

## 保存语义

当前配置跳过 epoch 0、1，从 epoch 2 开始每个 epoch 更新一次 optimizer。
因此完成 epoch 11 时 `global_step=10`，保存 `next_epoch=12`。
checkpoint 在 optimizer、清梯度、scheduler、长期 EMA 和 KL-reference EMA
全部更新之后生成，尚未消费下一轮 prompt，也尚未执行下一轮 eval。

每 10 个训练 step 保存一次，正常达到 `num_epochs` 时也保存最后一轮。
没有新增时限前主动退出机制；4h 到时若被杀，重做最近完整 checkpoint 之后的
通常最多 9 个已完成更新，以及被打断的当前轮。若在写 checkpoint 时中断，
也需重做触发这次保存的第 10 个更新。
暂不自动删除历史 checkpoint（`num_checkpoint_limit` 不控制这套保存器）。

保存内容：

- policy LoRA、KL-reference LoRA；AdamW 一阶/二阶矩及 step；LR scheduler；FP16 GradScaler。
- `global_step`、`next_epoch`、显式 rollout 游标、数据/配置/代码签名。
- 每个逻辑 rank 的 Python / NumPy / CPU Torch / CUDA RNG 和长期 EMA。
- 每个物理 rank 的环境 RNG、prompt 历史计数和 W&B run ID。
- 兼容原有推理入口的 `lora_non_ema/`、`lora_ema/` 导出；后者是逻辑 rank 0 的长期 EMA。

所有 rank 参与。先写唯一的临时目录，全部成功后发布 `complete.json` 并原子
重命名为 `checkpoint-N`。恢复检查文件列表和大小及 metadata 校验和。
不会覆盖已经存在的同 step checkpoint；若要从较早 checkpoint 分叉实验，
不要与原 run 并发写同一目录。

旧代码的 LoRA-only checkpoint 缺 optimizer/RNG/游标，不能变成精确 resume。
仍可通过 `train.lora_path` 做权重 warm start，但训练进度、optimizer 和 RNG 从头开始。

## W&B step 与中断重做

W&B 内部 `Step`、`train/global_step` 和图片的 step 使用同一个训练更新编号。
保留原有编号习惯：第一轮训练记录为 0，完成更新后 checkpoint 中的
`global_step` 为 1；eval 仍记录在评估时的训练编号，不改变评估时机。
同一步的 reward、eval、图片、loss 和耗时合并提交；图片在临时文件仍存在时
交给 SDK，epoch 末两套 EMA 更新完成后提交最后一条记录，再保存 checkpoint。

恢复沿用原 W&B run ID，在线模式要求 run 已存在。初始化后读取 W&B 的
下一条可写 step：例如 checkpoint 的 `global_step=100`，W&B 已记录到 107，
则照常重做 100～107 的训练和评估，但不重复上传这些编号的指标或图片；
从 108 开始继续上传。不能为了跳过日志而跳过训练、eval 或其 RNG 消耗。
如果 W&B 比 checkpoint 落后，允许从恢复后的训练编号继续，缺失记录不补造。

保留的重做区间日志来自中断前的轨迹；若重做有数值差异，这些日志与新权重轨迹
不一定完全一致。这是选择少改动、保留历史记录的折衷。
这里的提交是 SDK history 提交，不保证中断瞬间云端已经收到；以恢复时 W&B
实际返回的进度为准。离线模式不能提供云端续接保证。

不迁移旧版按 log 调用次数计数的历史记录。
训练代码签名不匹配的 checkpoint 会被一致性检查拒绝。

离线日志回归测试（无需 GPU 或云端上传，需要本机 socket）：

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_wandb_logging.py -v
```

## 为什么固定 32 条逻辑流

旧 4-node：32 ranks × 9 rollout batches × 6 images = 1728 images/epoch。
每个逻辑 rank 自己 shuffle 54 张图，再做 18 个 batch=3 的训练 microsteps。

新 4-node 每张卡负责一个逻辑 rank；新 2-node 的物理 rank `r` 负责 `r` 和 `r+16`，
分别沿用各自的采样、训练噪声、timestep、shuffle、eval RNG 和长期 EMA。
全局 reward/prompt 在计算 advantage 前重排为旧 32-rank 顺序。
不是把本卡两份样本混在一起 shuffle，也不是换成新的 hash seed 轨迹。

旧 sampler 已按 rank 切分，又被 Accelerate 包装，并存在预取/提前取批。
训练侧不再使用这个 loader，而是直接按显式编号取 prompt，兼容旧环境实际顺序：

```text
q = epoch * logical_rounds + rollout_round
sampler_seed = 42 + max(0, q - 3)
```

使用相同的 `randperm → 每条重复 24 次 → randperm → rank 切片`。
即保留旧 4-node 的三批延迟结果，但不再依赖 worker 队列来维持它。
这个兼容规则对应当前 Accelerate 1.9.0/PyTorch 2.6.0、worker=1 的原始训练 loader；
不是其他 loader 配置的通用公式。旧 2-node 的数据轨迹会改变为旧 4-node 的轨迹。
OCR/PickScore 的旧 4-node loader 使用同样的 sampler/batch/worker 设置，
相同的三批延迟规则通过实际 prepared-loader 回归测试；PickScore 每 epoch
使用 3 个逻辑 rollout rounds，而不是 Geneval/OCR 的 9 个。

为保留旧 4-node 的初始化后 RNG，2-node 会额外串行初始化对应逻辑 rank 的模型，
只保留其 RNG/长期 EMA，再释放临时模型；不会同时常驻两套完整 SD3 pipeline。
恢复时 checkpoint 会覆盖这些初始化状态。训练仍只使用一套共享 policy。

eval 同样按固定 32-rank 切片、padding、随机流及对应长期 EMA 执行，不再让物理
节点数改变下一轮训练的随机流。FP16、DDP bucket 重建和归约/累加顺序仍可能导致
权重的微小差异；不承诺长程权重逐 bit 相同。

## 验证入口

从 `advantage_weighted_matching/` 目录、激活主训练环境后运行：

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_resume.py -v
PYTHONPATH=. python -m unittest discover -s tests -p test_task_layout.py -v
```

GPU 回归脚本：

- `tests/distributed_resume_smoke.py`：小型 LoRA，真实 CUDA/DDP/FP16，8→8 和 8→4 恢复。
- `tests/train_entrypoint_smoke.py` / `tests/compare_entrypoint_smoke.py`：真实训练主入口、
  小型 SD3 transformer，替换外部文本/采样/reward 服务；验证 loss/KL、eval、
  完整 state 及逐批输入。
  入口测试可用 `--task geneval|ocr|pickscore` 选择布局；各任务使用独立
  输出目录，比较器读取保存的每逻辑 rank microbatch 数。
- `tests/sd35_rng_smoke.py`：真实 pretrained SD3.5，旧式 RNG 与新逻辑 RNG 的
  14-step SA-solver 采样对照。
