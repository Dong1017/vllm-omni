# Noisy PP on AR-Diffusion — Interface Draft

> 2026-09-17 · experimental 接口草案，供 noisy-chunk-pp（Dong1017）与
> noisy-chunk-pp-layer-step（ShengDev）双方对齐后动代码。
> 基线：`vllm_omni/experimental/ar_diffusion/`（AR-DiT runtime，vLLM paged KV 栈复用）。
> 目标：把 noisy chunk step-wise 调度从"两份实验分支里的专用实现"重构为
> USP2 之上的通用并行策略，掩盖 lingbot / minimax-based / wan-based 的模型差异。

## 0. 现状盘点（为什么要重做）

| 资产 | 位置 | 状态 |
|---|---|---|
| chunk 调度 + slot 排序 | 两分支各自的 `chunk_pipeline_parallel.py` | 分叉：Dong 版有 kv_source_policy 解耦 + producer 身份；Sheng 版有 world 泛化 carry 队列 + 去 gap/cond |
| KV 生命周期 | Dong 版 refcount / Sheng 版 last-use 淘汰 | 两套互斥；都基于 per-rank dict（`(task, producer)` 键） |
| 采样更新契约 | Dong 版 `causal_dmd.py`（高噪/边界/交接公式） | 已对齐 FastVideo 参考，可整体迁移 |
| AR-DiT KV 抽象 | `ar_diffusion/kv_cache/`（paged KV、named branches、session state） | **已成型的第三方**，本设计复用它而非重造 |
| 实测基线 | 2×H200: full 444s / serial-clean 273s / stepwise 145s (3.06×) | 同步传输 + 现 KV 机制下的数字，新架构需重测 |

## 1. 分层架构

noisy PP ≈ USP2 + PP 的组合。三层各自可独立演进，只通过类型化接口相接：

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1  NoisyPPScheduler          （纯函数，CPU 可测）        │
│   (chunk, step) 任务图 → slots[world]                        │
│   输入: world, chunks, steps, source_policy, window          │
│   不知道 KV、不知道模型。USP2 的 slot 语义 + noisy 依赖规则      │
├─────────────────────────────────────────────────────────────┤
│ Layer 2  NoisyKVManager           （ar_diffusion kv_cache 之上）│
│   clean branch + noisy branch 分池（named KV branches 复用）   │
│   版本生命周期: publish / fetch / retire，无全局 index 暴露     │
│   多 worker 并发访问；模型形态由 adapter 注册                   │
├─────────────────────────────────────────────────────────────┤
│ Layer 3  NoisyKVConnector         （异步传输）                 │
│   publish/fetch 返回 Future；ready-event 替代同步 wait        │
│   与层 2 的契约: fetch 前 manager 保证版本未 retire            │
└─────────────────────────────────────────────────────────────┘
        ↑ adapters/wan.py | lingbot.py | minimax.py 掩盖模型差异
```

## 2. Layer 1 — Scheduler 接口

```python
@dataclass(frozen=True)
class NoisyPPTask:
    chunk: int
    step: int          # step == num_steps 表示 clean pass（若启用）
    kind: Literal["denoise", "clean"]

@dataclass(frozen=True)
class NoisyPPSlot:
    index: int
    tasks: tuple[NoisyPPTask | None, ...]   # 长度 = world；None = 空槽
    reads: tuple[tuple[int, int], ...]      # 本槽各任务消费的 (chunk, version_kind) 摘要，仅调度元数据

def plan_noisy_pp(
    *,
    world: int,                 # PP 节点数，>=1（USP2 carry 队列语义）
    chunks: int,
    num_denoise_steps: int,
    enable_clean_pass: bool,    # KV 模式下每 chunk 尾部 t=0 context pass
    source_policy: Literal["latest", "clean"],   # latest=stepwise/replay 口径；clean=Self Forcing 口径
) -> list[NoisyPPSlot]:
    """纯函数。抛 ValueError 表示配置非法（如 clean+stepwise）。"""
```

设计决策：
- **调度与 KV 解耦**：scheduler 只产任务顺序与依赖；"读哪个版本"由层 2 在运行时解析
  （latest=执行时取最新已完成；clean=固定取 clean pass 版本）。Dong 版已验证两种
  policy 可共享同一 slot 计划（replay 对照测试）。
- **world 泛化**：采纳 Sheng 版 carry 队列（`carry = list(slot[:-1])`），单实现支持 1..N 节点。
- **依赖规则**：latest 策略下 (c, s) 依赖 (c, s-1)；clean 策略下 chunk c 依赖全部
  前继窗口的 clean pass 完成。serial/stepwise 是排序策略，不再是语义开关。

## 3. Layer 2 — KV Manager 接口

复用 `ar_diffusion/kv_cache`：clean 与 noisy 是**两个命名 KV branch**（`ARDiffusionKVBranchSpec`），
paged 存储与 slot 管理交给 vLLM 栈；本层只加 noisy 语义。

```python
class NoisyKVManager:
    def publish(self, task: NoisyPPTask, producer: str, kv: KVTensor) -> KVVersion: ...
    def fetch(self, task: NoisyPPTask, producer: str, sources: Sequence[VersionRef]) -> Future[KVTensor]: ...
    def retire(self, version: KVVersion) -> None: ...          # 引用归零后由消费者触发
    def resident_bytes(self) -> int: ...                        # profiling

@dataclass(frozen=True)
class VersionRef:
    chunk: int
    kind: Literal["latest", "clean"]     # 运行时解析，替代两分支各自硬编码
    producer: str                        # 双塔: "high"/"low"；单塔: "single"
```

设计决策：
- **clean/noisy 分池**：clean pass 版本进入 clean branch（长驻，窗口滑动淘汰）；
  denoise 版本进入 noisy branch（短期，消费者完成即 retire）。两种池的淘汰规则不同，
  这是"drop index 管理"的实质——index 变成 branch 内部实现，外部只见 handle。
- **producer 正交于 kind**：双塔模型 high/low 各有独立 clean/noisy 存储（Dong 版
  review #2 的整改成果，以 branch 命名表达：`{clean,noisy} × {high,low}`）。
- **retire 由 refcount 驱动**（保留 Dong 版机制；Sheng 的 last-use 是其特例：
  冻结 sources 的消费者计数在 plan 期算好）。跨 worker 的 retire 通知走层 3。
- **模型形态适配**：adapters 注册「如何从一次 forward 抽取 per-layer K/V」与
  「如何把 fetch 回的 K/V 喂给 attention」——wan 的 post-RoPE per-layer 布局是
  第一个 adapter；lingbot/minimax 差异不得泄漏到层 1/2。

## 4. Layer 3 — Connector 接口

```python
class NoisyKVConnector:
    def publish_async(self, version: KVVersion, tensor: KVTensor) -> Future[None]: ...
    def fetch_async(self, ref: VersionRef) -> Future[KVTensor]: ...
    # 契约: fetch 返回的 Future 在消费前 ready；retire 只能发生在
    # 所有已发出的 fetch Future 完成之后（由层 2 的引用计数保证）。
```

现状对照：两分支当前是同步 `isend_tensor_dict` + `handle.wait()`（每 slot 阻塞）。
异步化后传输与计算重叠的收益需单独量化（预期收益在 world≥4 或 KV 较大时显现；
2 节点下 P2P 仅 ~40MB/请求，可能不显著——先实现 API，收益另测）。

## 5. 与生产路径的关系

- `vllm_omni/diffusion/` 生产路径本轮**冻结不动**；wan2_2 pipeline 通过
  `engine_backend` 式开关选择实验运行时（对齐 `ARDiffusionEngine` 的选择机制）。
- 迁移清单（Dong → experimental）：`causal_dmd.py` 采样契约 → `adapters/wan.py`；
  kv_source_policy 校验 → `plan_noisy_pp`；`_dit_router` → wan adapter 的 tower 选择。
- 迁移清单（Sheng → experimental）：carry 队列 slot 计划 → `plan_noisy_pp`；
  `_resolve_temporal_chunks` 的请求推导 + 并行配置校验 → 调用侧。
- 两分支现有 112 测试迁移为：调度纯函数测试（CPU）+ KV manager 单测 + 2×H200 双模型
  冒烟（FastWan/CausalWan × serial-replay/serial-clean/stepwise）。

## 6. 复用映射（v2 修订：只写策略，不重建基础设施）

盘点后确认仓库已有完整基座，noisy_pp 是**薄策略层**：

| 能力 | 复用的现成实现 | noisy_pp 里只写 |
|---|---|---|
| KV 存储/paging/slot | `ar_diffusion/kv_cache/manager.py`（`ARDiffusionKVCache`，648 行，vLLM paged 栈封装）+ `paged.py` + `state.py`（多 branch session） | clean/noisy 分池策略（branch 命名 `clean_{producer}` / `noisy_{producer}`）、refcount retire、latest/clean 版本解析 |
| KV 传输 | vLLM `KVConnectorBase_V1` 生命周期 hook（`start_load_kv`/`wait_for_layer_load`/`save_kv_layer`/`get_finished`）；`vllm_omni/distributed/omni_connectors/kv_transfer_manager.py`（ZMQ rank-aware） | 版本寻址语义（VersionRef）、retire 安全门（`drained`）、stream-ordering 契约 |
| 传输组装方式 | `diffusion_kv/kv_connector.py`（Mooncake 经 `KVConnectorFactory`） | ——（参照模式） |
| tensor 布局 | `diffusion_kv/layout.py` / `paged.py` 的 `pool_write_chunk` | wan per-layer post-RoPE 视图映射（若 layout 覆盖则零新增） |
| 采样契约 | —— | `adapters/wan.py`（自 noisy-chunk-pp 分支平移：高噪/边界/交接公式 + 塔路由） |

因此 `kv_manager.py` 重写为 `ARDiffusionKVState/ARCache` 之上的策略类（无自建存储）；
`kv_connector.py` 声明 `NoisyKVTransfer` 适配协议（指向既有引擎），`LocalNoisyKVConnector`
覆盖 world==1 与 CPU 测试。

## 7. 已知风险

1. **异步传输正确性**：同步 wait 的"消费前到位"保证移到 Future 契约，ready 事件
   在 CUDA stream 上的语义要小心（event record ≠ 同步）。
2. **paged KV 形态匹配**：ar_diffusion 的 paged 栈面向 token 序列 KV；wan 的
   per-layer post-RoPE 张量布局需要 adapter 做视图映射，零拷贝能否达成待验证。
3. **cuda graph**：chunks==1 退化路径必须保持原生 PP + cuda graph 可用；
   noisy 路径的动态 slot 数可能阻止 graph capture——只对 chunk 路径禁用，不波及原生。
4. **性能基线**：3.06× 是同步传输 + dict KV 下测的；抽象层若引入 handle 间接寻址，
   需在 2×H200 复测，防止抽象税。
