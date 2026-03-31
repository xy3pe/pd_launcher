# Prefill Running 无法突破 1 的问题定位

## 问题现象

对 prefill 节点进行压测时，vLLM 日志中 `Running` 始终停留在 1（偶尔为 0），waiting 队列持续堆积：

```
Engine 000: Avg prompt throughput: 7234.3 tokens/s, Avg generation throughput: 1.4 tokens/s,
Running: 1 reqs, Waiting: 31 reqs, GPU KV cache usage: 1.9%
```

---

## 定位过程

### 阶段一：误判为 proxy 层问题

**初始方案**：在 proxy 层将 decode 阶段打桩（直接返回空响应），以为这样 prefill 节点就可以独立跑满。

**现象**：running 仍然为 1。

**原因分析**：proxy 的打桩只截断了 HTTP 请求，但 prefill vLLM 实例本身配置了 `kv_role: kv_producer`，在完成 prefill 计算后，engine 层仍会通过 Mooncake/HCCL 通道将 KV cache 推送给 decode 节点，并**同步等待对方 pull 确认**，才释放 KV block、将请求移出 running。

```
Proxy (stubbed)    Prefill Engine         Decode Engine
                        │── prefill ──────>│
                        │── KV transfer ──>│ ← decode 未收到 HTTP 请求
                        │<── 等待 ack ─────│   KV 无人消费
                        │    (卡住)        │
```

**结论**：proxy 层打桩 ≠ prefill engine 层隔离，两者在 KV transfer 通道上仍强耦合。

---

### 阶段二：去掉 KV connector，真正隔离 prefill

**修改**：
- `pd_service_ctl.py`：prefill-only 模式下，prefill 实例启动时不添加 `--kv-transfer-config`
- `pd_proxy.py`：prefill-only 模式下，proxy 直接将 prefill 的真实响应（`max_tokens=1`）返回客户端，不转发 decode

**去掉 KV connector 后的 vLLM 启动命令**：
```bash
vllm serve ... --enable-chunked-prefill --long-prefill-token-threshold 16384 --num-release-cpu-blocks 5000
# 无 --kv-transfer-config
```

**现象**：running 从 1 升到 2，但仍无法更高。

---

### 阶段三：排查 vllm-ascend 隐式限制

对 `/code/vllm-ascend` 源码进行分析，发现两个疑似限制点：

#### 疑似限制 1：`max_num_partial_prefills = 1`（实际无效）

`platform.py` 中强制将该值重置为 1：
```python
# platform.py:499
if getattr(vllm_config.scheduler_config, "max_num_partial_prefills", 1) != 1:
    logger.warning("Parameter '--max-num-partial-prefills' is optimized for ROCm. Resetting to default (1) for Ascend.")
    vllm_config.scheduler_config.max_num_partial_prefills = 1
```

看似硬限制并发 prefill 为 1，但验证 vLLM v1 调度器源码后发现：

```bash
grep "max_num_partial_prefills" vllm/v1/core/sched/scheduler.py
# → 无输出
```

**该字段只被 ROCm 的 attention backend 使用，标准 v1 调度器完全不读取此字段。Ascend 的 reset 是死代码，对调度无任何影响。**

#### 疑似限制 2：`long_prefill_token_threshold` 与 `max_num_batched_tokens` 比值（真正瓶颈）

调度器每步 token 分配逻辑（`patch_balance_schedule.py`）：

```python
# 对超过阈值的请求，限制本步处理的 token 数
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
num_new_tokens = min(num_new_tokens, token_budget)
```

每步 token 预算 = `max_num_batched_tokens`。
每个超长请求的 chunk 大小 = `long_prefill_token_threshold`。

**当时的参数**：
```
max_num_batched_tokens    = 32768
long_prefill_token_threshold = 16384
```

```
32768 / 16384 = 2  →  每步最多调度 2 个 chunk  →  Running = 2
```

对于 prompt 长度 > 16384 tokens 的请求，每个请求在一步内占用一个 16384-token slot，整个 budget 只够放 2 个，与观测完全吻合。

---

## 解决方案

调整 `long_prefill_token_threshold` 和 `max_num_batched_tokens` 的比值：

| 参数 | 原值 | 调整值 |
|------|------|--------|
| `max_num_batched_tokens` | 32768 | 65536 |
| `long_prefill_token_threshold` | 16384 | 1024 |

```
65536 / 1024 = 64  →  理论每步最多 64 个 chunk
```

实际受 KV cache 容量、请求数量等约束，最终 running 达到 33：

```
Engine 000: Avg prompt throughput: 2847.2 tokens/s, Avg generation throughput: 0.7 tokens/s,
Running: 33 reqs, Waiting: 0 reqs, GPU KV cache usage: 77.1%
```

waiting 队列清空，GPU KV cache 占用 77.1%，说明此时瓶颈已从调度转移到 **KV cache 容量**。

---

## 参数调优总结

| 参数 | 作用 | 调优建议 |
|------|------|----------|
| `max_num_batched_tokens` | 每步 token 预算总量，直接决定并发上限 | 根据显存和吞吐目标尽量调大 |
| `long_prefill_token_threshold` | 单步单请求最大 token 数（chunk size） | 调小可提升并发，但增加每条请求的完成步数 |
| `enable_chunked_prefill` | 是否开启分块调度 | prefill 压测时必须开启 |
| `max_num_seqs` | running 请求数上限 | 建议设置大于预期并发数（当前 256，足够） |

**核心公式**：

```
理论最大 running ≈ max_num_batched_tokens / long_prefill_token_threshold
实际 running 受限于：KV cache 容量、并发请求数、请求 prompt 长度分布
```

---

## 最终有效配置（prefill-only 压测）

```yaml
proxy:
  port: 8050
  prefill_only: true   # 不启 decode 节点，不挂 KV connector

vllm_defaults:
  max_model_len: 32768
  max_num_seqs: 256

prefill_defaults:
  enable_chunked_prefill: true
  max_num_batched_tokens: 65536
  long-prefill-token-threshold: 1024
  num-release-cpu-blocks: 5000
```
