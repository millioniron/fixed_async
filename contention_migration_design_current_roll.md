# 多 Agent 资源竞争场景迁移设计草案（面向当前 ROLL）

> 目标仓库: `<ROLL_REPO>`
> 参考文档:
> - `<ROLL_1201_REPO>/docs/contention_integration_plan_final.md`
> - `<ROLL_1201_REPO>/docs/contention_integration_plan.md`
> 产出性质: 面向当前 `ROLL` 代码架构的迁移设计草案，不直接复用 `ROLL_1201` 的实现假设
> 日期: 2026-03-30

---

## 1. 背景

`ROLL_1201` 中的 contention 方案文档，核心目标是将多 Agent 共享资源竞争场景接入 ROLL 的 agentic RL 训练流程，使策略在真正并发、存在竞争条件的环境中进行 self-play 学习。

但当前 `ROLL` 与文档假设存在明显架构漂移：

- 调度侧已经从 `RequestScheduler` 切换到 `RouterManager`
- `RolloutScheduler` 的初始化、挂起、恢复、退出协议已发生变化
- `AgenticPipeline` 的异步 old log probs / version lag 链路比旧文档更复杂
- 当前仓库中并不存在任何已落地的 contention 环境实现，只有设计文档

因此，本次工作不应定义为“把 `ROLL_1201` 代码搬到 `ROLL`”，而应定义为：

**以旧文档为需求来源，在当前 `ROLL` 架构上重新设计并落地 contention 集成。**

---

## 2. 现状结论

### 2.1 当前代码中的稳定基线

当前 `ROLL` 中与本次迁移直接相关的稳定入口如下：

- `roll/distributed/scheduler/rollout_scheduler.py`
- `roll/distributed/scheduler/router.py`
- `roll/pipeline/agentic/env_manager/traj_env_manager.py`
- `roll/pipeline/agentic/agentic_pipeline.py`
- `roll/pipeline/agentic/agentic_config.py`
- `roll/configs/base_config.py`

其中当前事实是：

- rollout 采样由 `RolloutScheduler` 统一驱动
- 推理由 `RouterManager` 管理并支持 `initialize / suspend / resume / shutdown`
- `traj_group_id` 仍按静态 `group_id` 语义生成
- `create_step` 已写入 rollout，用于异步版本相关统计和重算
- 当前代码已经新增了一批围绕 `traj_id` 的诊断日志，用于排查 batch 复制、重复 trajectory 和 TOPR 本地 batch 对齐问题

### 2.1.1 2026-03-31 代码状态补充

最近一轮代码改动已经落在以下几个点上：

- `roll/pipeline/agentic/env_manager/traj_env_manager.py`
  - 在 rollout 入队时打印 `traj_id / traj_group_id / group_id / episode_id / env_id / start_step`
- `roll/distributed/scheduler/rollout_scheduler.py`
  - 在 `get_batch()` 聚合每个 group rollout 时，统计 `traj_id` 重复情况
- `roll/pipeline/agentic/agentic_pipeline.py`
  - 在 `get_batch`、`adjust_batch`、actor 训练前增加 batch 级 `traj_id` 重复统计日志
- `roll/pipeline/agentic/agentic_actor_pg_worker.py`
  - 在 TOPR 路径增加本地 batch 对齐检查和 mismatch 错误日志
- `examples/fixed_async_offpolicy/tu_pipeline_env.yaml`
  - 当前实验配置已切到 `adv_estimator: reinforce`、`ratio_type: segment`、`pg_variant: topr`，并启用了 `interp_prox_loglinear`

这些改动的重要结论是：

- **当前仓库已经开始围绕 trajectory 重复/复制问题做诊断**
- **但这些改动仍然不是 contention/global-pool 功能实现本身**
- 因此本文档仍以“contention 尚未落地，只是基线代码已具备更强观测能力”为前提

### 2.2 当前缺失的能力

当前 `ROLL` 尚不存在以下 contention 组件：

- `ContentionEnv`
- `SharedScenario`
- `SharedEnvRegistry`
- `ScenarioDispatcher`
- `GlobalPoolGroupQueue`
- `GlobalPoolGroupQueueManager`
- `ContentionGroupFilter`

因此本迁移是新增能力，不是增量修补。

### 2.3 旧方案中仍然成立的核心思想

以下思想仍然成立，应保留：

- 训练目标是多 Agent 并行竞争，而不是单 Agent 模拟竞争
- 每个 Agent 独立输出 trajectory，训练时仍按单 trajectory 样本进入 PPO/GRPO
- GRPO 的归组应基于“同一竞争副本集合”的 rollout，而不是静态 worker group
- 多 Agent 协调应尽量下沉到环境层，不侵入模型推理逻辑
- 全局池比静态 group 更适合解决 episode 漂移问题

### 2.4 旧方案中需要重写的部分

以下内容不能直接照搬：

- 基于旧 `RequestScheduler` 的生命周期假设
- 将“兼容原 `GroupQueueManager` 接口”视为充分条件
- 对 async 模式的正确性证明中，未纳入当前 `old_log_probs_source`、`snapshot`、`version lag` 链路
- shutdown / abort 行为只考虑 group queue，不考虑 router inflight request

---

## 3. 设计目标

本次迁移目标分为四层。

### 3.1 功能目标

- 支持 3-8 个 Agent 在同一共享环境内竞争有限资源
- 支持真正并发的“谁先推理完谁先执行”
- 支持每个 Agent 独立生成 trajectory，并进入现有 agentic PPO/GRPO 训练
- 支持同一场景的 N 个副本用于 GRPO 方差缩减

### 3.2 训练语义目标

- 同一 GRPO group 内的 rollout 必须对应同一竞争场景分配语义
- `traj_group_id` 必须准确表达“应当一起归一化”的样本集合
- 不能依赖 dummy agent 填充
- 不能在 `async_generation_ratio > 0` 时发生 episode 漂移

### 3.3 工程目标

- 非 contention 环境保持原行为
- 推理路由逻辑尽量不改
- 对 `AgenticPipeline` 的侵入控制在配置和 batch 语义层
- 核心复杂度集中在 scheduler/env 层，不扩散到 actor/reward worker

### 3.4 可靠性目标

- shutdown 可解除所有阻塞等待
- env reset 失败时，能够完整回收 dispatcher / shared scenario 资源
- 过期 episode、被过滤 group、中断 rollout 都不会泄漏状态

---

## 4. 推荐总体方案

推荐继续采用旧文档中的主思路，但面向当前 `ROLL` 重写为：

**全局池调度 + 场景分配器 + 共享竞争环境**

分为三层：

1. 调度层：在当前 `GroupQueueManager` 基础上新增 global-pool 模式
2. 协调层：新增 `ScenarioDispatcher`，负责把不同 seed 的 group 组合成竞争场景
3. 环境层：新增 `ContentionEnv`、`SharedScenario`、`SharedEnvRegistry`

其中最重要的原则是：

**RouterManager 不感知 contention。**

也就是说，contention 不应改写推理路由逻辑；竞争协调只发生在 rollout 分配和 env step 层。

---

## 5. 目标架构

### 5.1 调度层

在当前 `roll/distributed/scheduler/rollout_scheduler.py` 中新增可选 global-pool 机制：

- 默认仍走现有 per-group `GroupQueueManager`
- contention 环境启用后，调度层切换为 global-pool episode 分配

推荐实现方式不是直接替换整个 `RolloutScheduler`，而是：

- 保留 `RolloutScheduler`
- 在其内部为 `env_output_queue` 增加模式开关
- 新增 `GlobalPoolGroupQueue` 和 `GlobalPoolGroupQueueManager`

目标语义：

- 所有 env_id 共享一组 episode slot
- 每个 episode slot 容量为 `group_size = N`
- 任意 N 个先拿到同一个 episode 的 env 构成一个 GRPO group

### 5.2 协调层

新增 Ray actor：

- `ScenarioDispatcher`

职责：

- 管理待分配 contention scenario 队列
- 按 seed 聚合属于同一 GRPO group 的 env
- 将多个不同 seed 的 group 凑成一个 M-agent 场景
- 为每个 seed 分配稳定的 `agent_index`
- 为同一 seed 的 N 个副本分配 `copy_index`
- 负责等待、唤醒、引用计数、释放和 shutdown unblock

### 5.3 环境层

新增组件：

- `ContentionEnv`
- `SharedScenario`
- `SharedEnvRegistry`

职责划分：

- `ContentionEnv` 作为单 Agent 外观，兼容现有 `TrajEnvManager`
- `SharedScenario` 作为单个竞争场景副本，持有共享状态和执行锁
- `SharedEnvRegistry` 管理 `(scenario_id, copy_index)` 到 `SharedScenario` 的映射

核心语义：

- 同一 `copy_index` 下的 M 个 Agent 共享一个 `SharedScenario`
- 不同 `copy_index` 表示同场景的不同独立副本
- 每个 Agent 的 `env.step()` 最终落到共享场景实例上，通过锁保证状态一致性

---

## 6. 与当前 ROLL 的接口衔接

### 6.1 RolloutScheduler

当前 `RolloutScheduler` 已绑定：

- `router_manager.initialize()`
- `router_manager.resume()`
- `router_manager.suspend()`
- `router_manager.shutdown()`

因此本次迁移不应创建第二套 rollout 生命周期。

设计要求：

- contention 只替换 `env_output_queue` 的 episode/group 组织方式
- `RolloutScheduler.get_batch()` 的入口、返回值和 router 生命周期保持不变
- global-pool queue 必须继续写入 `create_step`

### 6.2 TrajEnvManager

当前 `TrajEnvManager` 的关键假设是：

- `episode_id = output_queue.get_episode_id(group_id, env_id)`
- `seed = group_seed + episode_id`
- `traj_group_id = f"{tag}_{group_id}_{episode_id}_{group_seed}"`

为了支持 contention，需要改造为可插拔的 group 语义：

- 普通环境：保持现状
- contention 环境：从 output queue 获取 global-pool 元数据

推荐新增字段到 reset 结果或 env config 运行态中：

- `rollout_group_mode`: `static_group | global_pool`
- `episode_seed`
- `traj_group_id`
- `pool_episode_id`

其中 contention 模式下：

- `traj_group_id` 应使用 `tag_pool_{episode_id}_{base_seed}` 或等价稳定表达
- 不再依赖静态 `group_id`

### 6.3 AgenticPipeline

`AgenticPipeline` 不需要知道 contention 的场景细节，但需要接受其 batch 语义。

保持不变的点：

- 训练样本仍然是单 trajectory
- `group_by("traj_group_id")` 仍作为 GRPO/归组入口

需要确认的点：

- `compute_version_lag_metrics` 是否仍以 trajectory 为统计单位
- async 模式下 stale rollout 的 `create_step` 与 `traj_group_id` 是否会交叉污染
- reward normalization / advantage grouping 是否完全依赖 `traj_group_id`

结论：

- pipeline 预期只需少量兼容，不应承载 contention 主逻辑

### 6.4 配置层

建议在 `EnvManagerConfig` 或 `custom_envs[tag]` 中增加显式开关，而不是隐式按 env 名识别。

推荐新增配置：

```yaml
contention:
  enabled: true
  use_global_pool: true
  min_agents_per_scenario: 3
  max_agents_per_scenario: 8
  scenario_dispatcher_cls: roll.pipeline.agentic.env.contention.dispatcher.ScenarioDispatcher
  shared_registry_cls: roll.pipeline.agentic.env.contention.shared_env.SharedEnvRegistry
```

同时建议增加启动校验：

- contention 开启时，必须启用 `use_global_pool`
- contention 开启时，`group_size > 1`
- contention 开启时，训练 grouping 必须允许按 `traj_group_id`

---

## 7. 新增组件设计

### 7.1 GlobalPoolGroupQueue

建议新增文件：

- `roll/distributed/scheduler/global_pool_queue.py`

职责：

- 管理全局共享 episode slot
- 每个 slot 需要记录：
  - `episode_id`
  - `create_step`
  - `capacity = group_size`
  - `running_rollouts`
  - `rollouts`

要求：

- `advance_step(step)` 为全局新增 K 个 episode slot
- `get_episode_id(env_id)` 从全局按顺序分配 slot
- `put(episode_id, start_step, rollout)` 仍按 episode 聚合
- `get_batch(batch_size)` 按 episode 消费完整 group
- `shutdown()` 必须唤醒所有等待中的 `get_episode_id / get_batch`

### 7.2 GlobalPoolGroupQueueManager

建议新增文件：

- `roll/distributed/scheduler/global_pool_queue_manager.py`

职责：

- 对外保持和 `GroupQueueManager` 近似一致的 actor 接口
- 在 contention 模式下，忽略静态 `group_id`
- 为 `TrajEnvManager` 提供 episode 分配所需的附加元数据

推荐接口：

```python
async def get_episode_id(self, group_id, env_id=None) -> Optional[int]
async def get_episode_info(self, group_id, env_id=None) -> Optional[dict]
def put(self, group_id, episode_id, start_step, rollout, env_id=None)
async def get_batch(self, batch_size, step=None)
def advance_step(self, step)
def shutdown(self)
```

说明：

- 为减少对旧代码侵入，可以先保留 `get_episode_id`
- contention 路径新增 `get_episode_info`
- 普通环境继续只用 `get_episode_id`

### 7.3 ScenarioDispatcher

建议新增文件：

- `roll/pipeline/agentic/env/contention/dispatcher.py`

建议核心状态：

- `scenario_queue`
- `assignment_cache: seed -> CachedAssignment`
- `waiting_scenarios`
- `shutdown_flag`

`CachedAssignment` 建议包含：

- `scenario_id`
- `agent_index`
- `copy_count`
- `ready_event`
- `ref_count`
- `released_count`
- `scenario_data`

推荐接口：

```python
async def join(self, seed: int, env_id: int) -> Assignment
async def release(self, seed: int, env_id: int)
async def shutdown(self)
```

关键约束：

- 同 seed 的缓存命中路径也必须等待 `ready_event`
- `shutdown()` 必须使所有阻塞中的 `join()` 立即退出
- 当同 seed 的 N 个 env 都 `release` 后，才允许删除 cache

### 7.4 SharedScenario / SharedEnvRegistry

建议新增文件：

- `roll/pipeline/agentic/env/contention/shared_env.py`

`SharedScenario` 职责：

- 保存共享环境状态
- 串行化对环境状态的实际写入
- 跟踪每个 agent 的完成状态

`SharedEnvRegistry` 职责：

- 提供 `get_or_create(scenario_id, copy_index)`
- 提供引用计数和回收
- 防止重复创建同一副本

### 7.5 ContentionEnv

建议新增文件：

- `roll/pipeline/agentic/env/contention/contention_env.py`

要求：

- 对 `TrajEnvManager` 看起来仍是标准 gem 环境
- `reset(seed)` 内部执行 dispatcher join
- `step(action)` 将动作提交到共享场景
- `close()` 或 cleanup 路径完整释放 dispatcher / shared scenario 引用

---

## 8. 关键数据语义

### 8.1 seed 语义

普通模式：

- `group_seed = base_seed + static_group_id`
- `episode_seed = group_seed + episode_id`

contention global-pool 模式：

- 所有 env 共用一个 `base_seed`
- `episode_seed = base_seed + pool_episode_id`

要求：

- 同一 GRPO group 的所有 rollout 具有相同 `episode_seed`
- 不同 GRPO group 的 seed 相互独立

### 8.2 traj_group_id 语义

普通模式保持：

```python
f"{tag}_{group_id}_{episode_id}_{group_seed}"
```

contention 模式建议改为：

```python
f"{tag}_pool_{episode_id}_{base_seed}"
```

要求：

- 同一个 pool episode 的 N 个 rollout 完全一致
- 下游 `group_by("traj_group_id")` 无需修改

### 8.3 create_step 语义

`create_step` 仍由 queue 层写入，不应由 env 层手工构造。

要求：

- global-pool 模式也必须准确写入 `create_step`
- 过期 episode 被淘汰后，旧 rollout 若晚到，必须被安全忽略

---

## 9. 实施阶段

建议分三阶段实施。

### 阶段 1：全局池调度落地

目标：

- 在不引入 contention env 的情况下，先让 current ROLL 支持 global-pool group 语义

改动范围：

- `roll/distributed/scheduler/rollout_scheduler.py`
- 新增 `global_pool_queue.py`
- 新增 `global_pool_queue_manager.py`
- `traj_env_manager.py` 小幅兼容
- 单元测试

验收标准：

- 普通环境行为不变
- contention 开关开启时，`traj_group_id` 可正确按 pool episode 生成
- `get_batch()` 返回的 group 保持完整

### 阶段 2：场景协调与共享环境

目标：

- 接入 `ScenarioDispatcher + SharedScenario + ContentionEnv`

改动范围：

- 新增 `roll/pipeline/agentic/env/contention/` 目录
- 环境注册
- 配置入口
- reset/cleanup 异常路径处理

验收标准：

- 多 Agent 可在同一共享场景中竞争
- 同 seed 的 N 个副本形成独立拷贝
- shutdown 时无 join 卡死

### 阶段 3：训练链路验证

目标：

- 验证 async 训练、GRPO 归组、过滤和清理逻辑

改动范围：

- 过滤器
- 训练侧批处理兼容
- 指标与测试

验收标准：

- async 模式下无 episode 漂移
- version lag 统计可运行
- old log probs 重算与 `traj_group_id` 无冲突

---

## 10. 风险点

### 10.1 shutdown 死锁

风险：

- env 线程阻塞在 `ScenarioDispatcher.join()`
- router 已 shutdown，但 env 仍未解除阻塞

要求：

- dispatcher 必须有显式 `shutdown()`
- `RolloutScheduler.shutdown()` 在停止 env queue 时，要同时触发 dispatcher 退出

### 10.2 env reset 失败后的资源泄漏

风险：

- join 完成后 reset 失败，导致 assignment cache / shared scenario 引用未释放

要求：

- contention env manager 或 contention env 自身必须具备 reset-failed cleanup

### 10.3 过期 episode 与 stale rollout

风险：

- async 模式下 episode 可能过期
- 迟到 rollout 仍可能上报到 queue

要求：

- global-pool queue 沿用当前“过期 episode 忽略晚到 rollout”策略
- dispatcher / shared registry 也要有对应回收语义

### 10.4 traj_group_id 与 create_step 的交叉语义

风险：

- 同一 `traj_group_id` 的样本如果来自不一致的 stale 版本，可能影响统计解释

要求：

- 明确 `traj_group_id` 只表达归组，不表达版本
- 版本信息继续仅由 `create_step` / segment infer step 承担

### 10.5 对现有环境的回归影响

风险：

- 若直接修改 `GroupQueueManager` 主路径，可能影响 travel / tu_pipeline 等现有环境

要求：

- 一切 contention 行为必须显式开关控制
- 默认路径完全不变

---

## 11. 测试计划

至少补以下测试。

### 11.1 调度层测试

- global-pool `get_episode_id()` 能按容量形成 episode group
- `put()` / `get_batch()` 能正确按 episode 聚合
- async 模式下过期 episode 会被清理
- shutdown 能唤醒等待中的 `get_episode_id / get_batch`

### 11.2 协调层测试

- `ScenarioDispatcher.join()` 能将不同 seed 正确凑成 M-agent 场景
- 同 seed 的重复 join 命中缓存但仍等待 ready
- `release()` 引用计数正确
- shutdown 能解除阻塞中的 join

### 11.3 环境层测试

- 同一 copy 内多个 agent 共享状态
- 不同 copy 间状态相互独立
- “谁先推理完谁先执行”能反映到资源竞争结果
- reset 失败时资源被回收

### 11.4 训练链路测试

- contention 模式下 `traj_group_id` 归组正确
- async 模式下 `create_step` 保持正确
- `group_by("traj_group_id")` 不需要额外改动
- reward normalization / advantage 计算在 contention 数据上可运行

---

## 12. 推荐文件改动清单

### 12.1 新增文件

- `roll/distributed/scheduler/global_pool_queue.py`
- `roll/distributed/scheduler/global_pool_queue_manager.py`
- `roll/pipeline/agentic/env/contention/__init__.py`
- `roll/pipeline/agentic/env/contention/dispatcher.py`
- `roll/pipeline/agentic/env/contention/shared_env.py`
- `roll/pipeline/agentic/env/contention/contention_env.py`
- `roll/pipeline/agentic/env/contention/contention_env_manager.py`

### 12.2 修改文件

- `roll/distributed/scheduler/rollout_scheduler.py`
- `roll/pipeline/agentic/env_manager/traj_env_manager.py`
- `roll/pipeline/agentic/agentic_config.py`
- `roll/pipeline/agentic/env/__init__.py`
- 视需要修改：
  - `roll/pipeline/agentic/agentic_pipeline.py`
  - `tests/agentic/env_manager/*`
  - `tests/agentic/rollout/*`

---

## 13. 最小落地建议

如果希望降低风险，推荐按下面的最小可交付顺序推进：

1. 先只做 global-pool queue，不做 contention env
2. 验证 `traj_group_id` 新语义在当前训练链路中可跑通
3. 再补 `ScenarioDispatcher`
4. 最后接入 `ContentionEnv` 和共享状态执行

这样可以把问题切成三段：

- 调度正确性
- 场景分配正确性
- 竞争执行正确性

每段都可以独立验证，避免一次性把 scheduler、env、pipeline 三层同时改坏。

---

## 14. 最终建议

本次 contention 迁移应以当前 `ROLL` 为唯一实现基线。

`ROLL_1201` 的最终版文档可以继续作为需求和思路来源，但不应再作为接口级实现蓝图。真正落地时，应以当前 `RouterManager + RolloutScheduler + TrajEnvManager + AgenticPipeline` 的组合为准重新设计。

下一步建议直接进入“阶段 1：全局池调度落地”，先输出更细的接口设计和 patch 计划，再开始代码实现。
