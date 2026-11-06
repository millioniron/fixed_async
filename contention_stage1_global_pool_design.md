# Contention 迁移阶段 1 细化设计

> 范围: 仅覆盖“全局池调度落地”，不包含 `ScenarioDispatcher / ContentionEnv / SharedScenario`
> 目标: 在当前 `ROLL` 中先引入 global-pool GRPO 分组语义，并保持普通环境路径不变
> 日期: 2026-03-31

---

## 1. 阶段目标

阶段 1 只解决一个问题：

**让当前 `ROLL` 在不引入 contention 环境实现的前提下，具备 global-pool episode 分配能力。**

这一步的意义是先打通下面三件事：

- queue 层可以不依赖静态 `group_id` 形成 GRPO group
- `TrajEnvManager` 可以消费“全局池 episode 元数据”
- 下游训练仍然只依赖 `traj_group_id`，无需修改 PPO/GRPO 主流程

本阶段不做：

- 多 Agent 场景配对
- 共享状态竞争执行
- dispatcher / shared registry

---

## 2. 当前约束

### 2.1 不能破坏的现有行为

当前稳定行为如下，必须保持：

- 普通环境仍按静态 `group_id` 使用 `GroupQueueManager`
- `RolloutScheduler` 继续只管理一个 `env_output_queue`
- `RouterManager` 生命周期不感知 global-pool
- `TrajEnvManager.run_rollout_loop()` 仍输出单 trajectory rollout
- `AgenticPipeline` 继续只按 `traj_group_id` 归组

### 2.1.1 当前代码已新增的诊断能力

在开始阶段 1 实现前，需要把当前已落地的诊断逻辑视为既有基线，而不是临时噪音：

- `TrajEnvManager` 已在 rollout 入队前打印 `traj_id / traj_group_id / group_id / episode_id / env_id / start_step`
- `GroupQueueManager.get_batch()` 已增加 group 内 `traj_id` 重复统计
- `AgenticPipeline` 已在 rollout 收集、`adjust_batch` 前后、actor 训练前打印 batch 级 `traj_id` 统计
- `agentic_actor_pg_worker.ActorWorker` 已在 TOPR 路径中检查 `scores` 与 `log_probs` 的 batch 对齐

阶段 1 实现时，这些日志不应被删掉；它们会成为验证 global-pool 是否引入 trajectory 重复或 batch 失配的重要观测面。

### 2.2 当前代码的关键锚点

本阶段改动要围绕这些位置展开：

- `roll/pipeline/agentic/agentic_config.py:360`
  当前 `make_env_configs()` 用 `env_id // final_group_size` 固定生成 `group_id`
- `roll/distributed/scheduler/rollout_scheduler.py:358`
  当前 `GroupQueueManager` 是 per-group queue actor
- `roll/distributed/scheduler/rollout_scheduler.py:429`
  当前 env 通过 `get_episode_id(group_id, env_id)` 获取 episode
- `roll/pipeline/agentic/env_manager/traj_env_manager.py:145`
  当前 `reset()` 只知道 `episode_id`
- `roll/pipeline/agentic/env_manager/traj_env_manager.py:129`
  当前 `traj_group_id` 绑定静态 `group_id`

---

## 3. 设计原则

### 3.1 不替换 RolloutScheduler

不建议改成“另一套 rollout scheduler”。当前 `RolloutScheduler` 已经与 `RouterManager` 生命周期深度绑定，阶段 1 只替换其中的 queue 组织方式。

### 3.2 不让普通路径承担条件分支复杂度

global-pool 必须显式开关控制，默认路径不变。普通环境不应该在运行时额外承担复杂逻辑。

### 3.3 保持旧接口可用，增量补充新接口

现有 env manager 和测试大量依赖：

```python
get_episode_id(group_id, env_id=None)
put(group_id, episode_id, start_step, rollout, env_id=None)
get_batch(batch_size, current_step)
```

因此阶段 1 不应强推接口替换，而应在兼容旧接口的基础上增加：

```python
get_episode_info(group_id, env_id=None)
```

---

## 4. 配置设计

### 4.1 推荐新增配置

建议加到 `EnvManagerConfig`：

```python
use_global_pool: bool = False
global_pool_mode: Literal["disabled", "grpo"] = "disabled"
```

含义：

- `use_global_pool=False`：保持现状
- `use_global_pool=True`：启用全局池 episode 分配
- `global_pool_mode="grpo"`：明确当前只支持以 `group_size` 作为 pool slot 容量

### 4.2 配置约束

当 `use_global_pool=True` 时，建议增加启动校验：

- `group_size >= 2`
- `num_env_groups >= 1`
- `max_env_num_per_worker > 0`
- `group_size_redundancy == 0`

阶段 1 不支持 redundancy，原因很直接：

- 静态 `group_size_redundancy` 是 per-group 语义
- global-pool 的目标就是“真实 N 条 rollout 形成组”
- redundancy 会把“组容量”和“收敛容量”重新混淆

### 4.3 是否修改 make_env_configs

建议阶段 1 **不改** `make_env_configs()` 的 `group_id` 生成逻辑。

原因：

- worker 分片、日志、旧测试都依赖这个字段存在
- global-pool 模式下可以把 `group_id` 视为“兼容字段”，不再作为真实分组依据

结论：

- `env_config["group_id"]` 继续存在
- 但 queue 层在 global-pool 模式下忽略它

---

## 5. Queue 层设计

### 5.1 目标

把当前的：

- per-group `GroupQueue`

扩展为：

- 普通模式：`group_id -> GroupQueue`
- global-pool 模式：单例 `GlobalPoolGroupQueue`

### 5.2 建议新增对象

建议新增两个类：

```python
class GlobalPoolEpisodeData:
    episode_id: int
    create_step: int
    running_rollouts: int
    rollouts: list[DataProto]

class GlobalPoolGroupQueue:
    ...
```

其语义应接近当前 `GroupQueue`，但去掉静态 `group_id` 依赖。

### 5.3 GlobalPoolGroupQueue 核心状态

建议状态：

```python
self.group_size
self.max_traj_per_env
self.async_generation_ratio
self.current_step
self.next_episode_id
self.episodes: Dict[int, GlobalPoolEpisodeData]
self.progress
self.complete
self.quit
```

注意点：

- `episodes` 是全局共享，而不是 per-group
- 每个 episode 的容量就是 `group_size`
- episode 被完整收齐后，仍按 episode_id 顺序消费

### 5.4 advance_step 语义

建议保持与当前 `GroupQueue.advance_step()` 等价的 async 语义：

- 初次进入时，如果 `async_generation_ratio > 0`，额外预创建多步 episode
- 后续 step 推进时，移除超出 async 容忍窗口的旧 episode
- 为当前 step 追加新 episode slot

差别只在于：

- 当前是“每个静态 group 各 advance 一次”
- global-pool 是“全局 episode 池 advance 一次，但一次创建 `num_env_groups` 个 slot”

建议新增参数：

```python
num_episode_slots_per_step = env_manager_config.num_env_groups
```

理由：

- 这样可以保持每步总目标组数与原先一致

### 5.5 get_episode_info 语义

阶段 1 推荐让 queue 层直接返回完整 episode 元信息，而不只返回 `episode_id`：

```python
{
    "episode_id": int,
    "create_step": int,
    "pool_mode": "global_pool",
    "base_seed": int,
}
```

普通模式可以返回：

```python
{
    "episode_id": int,
    "create_step": int,
    "pool_mode": "static_group",
}
```

### 5.6 get_episode_id 兼容策略

保留旧接口：

```python
async def get_episode_id(self, group_id, env_id=None):
    info = await self.get_episode_info(group_id, env_id)
    return None if info is None else info["episode_id"]
```

这样现有 env manager / 测试可以继续跑。

### 5.7 put 语义

普通模式保持不变：

- 仍写入对应 `group_id` 的 `GroupQueue`

global-pool 模式下：

- 忽略 `group_id`
- 按 `episode_id` 写入 `GlobalPoolGroupQueue`

同时继续注入：

- `create_step`

### 5.8 get_batch 语义

普通模式保持现状：

- 从各个静态 group queue 收集完整组

global-pool 模式下：

- 从全局 episode 池直接取完整 episode group
- 每个 episode group 视为一个 GRPO group

关键点：

- `current_step - create_step > async_generation_ratio` 的丢弃逻辑必须保留
- `batch_size` 语义不变，仍是 trajectory 数量而不是 episode 数量

---

## 6. GroupQueueManager 设计

### 6.1 建议方式

阶段 1 不建议新增一个完全独立的 actor 文件后立刻切换。更稳妥的方式是：

- 先在现有 `GroupQueueManager` 内增加 global-pool 模式
- 等阶段 1 跑通后，再决定是否抽独立文件

原因：

- 现有 `RolloutScheduler` 和测试已经直接引用 `GroupQueueManager`
- 先在现有 actor 上加模式，改动面最小

### 6.2 建议内部结构

当前：

```python
self.group_queue: Dict[int, GroupQueue]
```

改为：

```python
self.use_global_pool: bool
self.group_queue: Dict[int, GroupQueue]
self.global_pool_queue: Optional[GlobalPoolGroupQueue]
```

### 6.3 各方法行为

#### `advance_step(step)`

- 普通模式：遍历 `group_queue.values()`
- global-pool：调用 `global_pool_queue.advance_step(step)`

#### `get_episode_info(group_id, env_id=None)`

- 普通模式：路由到对应 `GroupQueue`
- global-pool：忽略 `group_id`，从全局池分配

#### `put(group_id, episode_id, start_step, rollout, env_id=None)`

- 普通模式：按 `group_id`
- global-pool：按 `episode_id`

#### `get_batch(batch_size, current_step)`

- 普通模式：维持当前实现
- global-pool：只监听 `global_pool_queue.get()`

### 6.4 EnvActivityMonitor 兼容

阶段 1 推荐最小兼容，不彻底重写 monitor。

建议：

- 普通模式：保持 `(group_id, env_id)` 键
- global-pool 模式：monitor 仍记录 `(group_id, env_id)`，但日志上允许 group_id 仅作兼容展示

这不是最终最优设计，但足够支撑阶段 1。

---

## 7. TrajEnvManager 设计

### 7.1 当前问题

`TrajEnvManager.reset()` 只拿到 `episode_id`，无法判断：

- 当前是否启用 global-pool
- 应该如何计算 seed
- `traj_group_id` 应如何生成

### 7.2 推荐改法

在 `reset()` 中优先走新接口：

```python
episode_info = ray.get(self.output_queue.get_episode_info.remote(...))
```

返回值为空则退出。

### 7.3 运行态字段

建议给 `TrajEnvManager` 增加：

```python
self.episode_info = None
self.rollout_group_mode = "static_group"
self.base_seed = None
```

### 7.4 seed 计算

普通模式保持：

```python
seed = self.group_seed + self.episode_id
```

global-pool 模式改为：

```python
seed = episode_info["base_seed"] + episode_info["episode_id"]
```

注意：

- 这里的 `base_seed` 应由 queue actor 在启动 rollout loop 时固定
- 不应继续叠加静态 `group_seed`

### 7.5 traj_group_id 生成

建议提取一个小函数：

```python
def build_traj_group_id(self) -> str:
    if self.rollout_group_mode == "global_pool":
        return f"{self.rollout_cache.tag}_pool_{self.episode_id}_{self.base_seed}"
    return f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
```

这样：

- 普通环境零行为变化
- global-pool 有单一明确入口

同时要保留当前已经存在的 enqueue 日志，并在 global-pool 模式下让日志字段继续可读：

- `traj_id`
- `traj_group_id`
- `episode_id`
- `env_id`
- `start_step`

如果进入 global-pool 模式，建议把日志中的 `group_id` 继续保留为兼容字段，但不要再把它解释为真实 GRPO 分组来源。

### 7.6 rollout_cache 是否修改

阶段 1 不建议改 `RolloutCache` dataclass 结构。

理由：

- 当前只需要 env manager 内部能生成正确 `traj_group_id`
- contention 场景级元数据暂时不需要写进 rollout

若需要调试，可先附加到 `non_tensor_batch`：

- `pool_episode_id`
- `rollout_group_mode`

---

## 8. RolloutScheduler 改动边界

阶段 1 对 `RolloutScheduler` 的目标是“感知更少”。

建议不改：

- `initialize()`
- `shutdown()`
- `suspend()`
- `router_manager.*`

只让它继续依赖：

```python
self.env_output_queue.advance_step(...)
self.env_output_queue.get_batch(...)
```

如果 `GroupQueueManager` 内部已经做了 global-pool 分流，则 `RolloutScheduler` 无需知道更多细节。

这是阶段 1 最重要的隔离策略。

---

## 9. 最小接口草案

### 9.1 GroupQueueManager

```python
@ray.remote
class GroupQueueManager:
    def __init__(self, config, env_manager_config, mode):
        ...

    async def get_episode_id(self, group_id, env_id=None) -> Optional[int]:
        ...

    async def get_episode_info(self, group_id, env_id=None) -> Optional[dict]:
        ...

    def put(self, group_id, episode_id, start_step, rollout, env_id=None):
        ...

    async def get_batch(self, batch_size, current_step) -> list[DataProto]:
        ...

    def advance_step(self, step):
        ...

    def shutdown(self):
        ...
```

### 9.2 episode_info 结构

```python
{
    "episode_id": int,
    "create_step": int,
    "rollout_group_mode": "static_group" | "global_pool",
    "base_seed": Optional[int],
}
```

### 9.3 TrajEnvManager

```python
def reset(self) -> RolloutCache:
    episode_info = ray.get(self.output_queue.get_episode_info.remote(
        self.env_config["group_id"],
        self.env_config["env_id"],
    ))
    ...
```

---

## 10. 建议实现顺序

### 步骤 1

在 `EnvManagerConfig` 增加：

- `use_global_pool`
- `global_pool_mode`

并加配置校验。

### 步骤 2

在 `GroupQueueManager` 内增加 mode 分流，但先不抽新文件：

- 添加 `self.use_global_pool`
- 添加 `self.global_pool_queue`
- 添加 `get_episode_info`

### 步骤 3

补 `GlobalPoolGroupQueue` 实现，并复用当前 `GroupQueue` 的 async 过期清理逻辑。

### 步骤 4

改 `TrajEnvManager.reset()` 和 `traj_group_id` 构造逻辑。

### 步骤 5

补单元测试，确认普通路径和 global-pool 路径都通过。

---

## 11. 测试细化

### 11.1 新增 queue 层测试

建议新增测试覆盖：

- global-pool 模式下，连续 `N` 次 `get_episode_info()` 返回同一 `episode_id`
- 第 `N+1` 次分配到下一 `episode_id`
- 同一 step 下创建 `num_env_groups` 个 episode slot
- `put()` 收满 `group_size` 后可被 `get_batch()` 消费
- async 模式下过期 episode 被丢弃

### 11.2 TrajEnvManager 测试

建议新增或扩展现有测试：

- static 模式下 `traj_group_id` 不变
- global-pool 模式下 `traj_group_id` 改为 `tag_pool_episode_baseSeed`
- global-pool 模式下 seed 计算不再使用 `group_seed`

### 11.3 回归测试

必须保证以下旧测试不被破坏：

- `tests/agentic/rollout/test_rollout_scheduler.py`
- `tests/agentic/env_manager/test_traj_env_manager.py`
- `tests/distributed/scheduler/test_rollout_scheduler.py`

必要时可为旧测试中的 mock config 补默认字段：

- `use_global_pool = False`
- `global_pool_mode = "disabled"`

### 11.4 诊断日志验证

阶段 1 虽然不是日志功能改动，但因为当前代码已经强依赖这些观测信息排查 batch 问题，建议在联调时额外确认：

- global-pool 模式下 enqueue 日志中的 `traj_group_id` 已切到 pool 语义
- scheduler group 聚合日志中不出现意外的跨 episode 重复 `traj_id`
- `adjust_batch` 前后的 duplicate `traj_id` 统计符合预期
- 若使用 TOPR，`scores` 与 `log_probs_batch` 的 batch size 不出现失配

---

## 12. 风险与决策

### 12.1 风险：先在现有 GroupQueueManager 内加逻辑会让文件继续变重

结论：

- 这是阶段 1 可接受成本
- 先求兼容落地，再考虑阶段 2 时抽象重构

### 12.2 风险：episode_info 新接口会影响 mock / 旧调用

结论：

- 保留 `get_episode_id` 即可
- 旧代码不需要同步升级

### 12.3 风险：global-pool 仍然保留静态 group_id 字段会造成语义混淆

结论：

- 阶段 1 接受这种“兼容字段”存在
- 真正的分组语义只看 `rollout_group_mode + traj_group_id`

---

## 13. 阶段 1 完成判定

满足以下条件即可认为阶段 1 完成：

1. 当前 `ROLL` 能通过配置开启 global-pool 调度
2. `TrajEnvManager` 能在 global-pool 模式下生成正确 seed 和 `traj_group_id`
3. `RolloutScheduler` 与 `RouterManager` 不需要修改生命周期
4. 普通环境路径行为不变
5. 相关 queue/env manager 测试通过

---

## 14. 阶段 1 之后的接口冻结建议

如果阶段 1 跑通，建议将以下接口视为阶段 2 的稳定基线：

- `EnvManagerConfig.use_global_pool`
- `GroupQueueManager.get_episode_info()`
- `episode_info["rollout_group_mode"]`
- `TrajEnvManager.build_traj_group_id()`

阶段 2 的 `ScenarioDispatcher` 和 `ContentionEnv` 应建立在这组接口之上，而不要重新改写 queue 与 env manager 的基本交互。
