"""adaptive_throttle.py — gateway nginx 限速 adaptive throttle.

真因背景 (2026-06-02 用户拍板, benchmark 实证):
- gateway 网关用 nginx limit_req 模块按 RPS/RPM 限速 (429 body 是 nginx 错误页)
- 不是 LLM 厂商限流, 也不是 gateway 额度耗尽
- 等几秒钟就重置 → token bucket + circuit breaker 标准做法

3 件套设计:
  1. TokenBucket 限速 (令牌桶, RPS 上限)
  2. CircuitBreaker 熔断 (连续 429 → 全局暂停 backoff)
  3. AdaptiveRPS 自适应 (撞 429 降 RPS, 长时间无 429 升 RPS)

判断 "速率限 vs 额度限":
  - 速率限 (b): 429 body 含 nginx (gateway 网关), backoff 后能恢复
  - 额度限 (a): 429 body 含 "quota" / "credit" 或 余额查询 < threshold → raise AccountExhaustedError 立刻停

用法:
    from adaptive_throttle import GLOBAL_THROTTLE
    GLOBAL_THROTTLE.acquire()  # 阻塞直到能发请求
    try:
        response = call_llm(...)
        GLOBAL_THROTTLE.on_success()
    except RateLimitException as e:
        GLOBAL_THROTTLE.on_429(retry_after=e.retry_after)
        # retry 逻辑
"""
from __future__ import annotations
import time
import threading
import json
import os
import urllib.request
import urllib.error
from typing import Optional


class AccountExhaustedError(Exception):
    """额度真耗尽 — 跟速率限不同, 必须停 + 充值."""


class AdaptiveThrottle:
    """Token bucket + Circuit breaker + Adaptive RPS.

    Thread-safe. 全局单例 (GLOBAL_THROTTLE).
    """

    def __init__(
        self,
        initial_rps: float = 5.0,        # 起始 RPS (从 benchmark 实测调)
        min_rps: float = 1.0,            # RPS 下限 (再低就基本停转)
        max_rps: float = 50.0,           # RPS 上限 (再高肯定撞 nginx)
        backoff_base: float = 2.0,       # 429 后 wait = base ** consecutive_429
        backoff_max_s: float = 60.0,     # 单次 backoff 上限
        circuit_429_threshold: int = 5,  # 连续 N 个 429 → 触发熔断
        circuit_pause_s: float = 120.0,  # 熔断后暂停时长
        downscale_factor: float = 0.7,   # 429 后 RPS *= 0.7
        upscale_factor: float = 1.05,    # 60s 无 429 → RPS *= 1.05
        upscale_check_s: float = 60.0,   # 升 RPS 检查窗口
    ):
        self.tokens = initial_rps
        self.capacity = initial_rps
        self.refill_rate = initial_rps
        self.last_refill = time.time()
        self.min_rps = min_rps
        self.max_rps = max_rps

        self.backoff_base = backoff_base
        self.backoff_max_s = backoff_max_s
        self.circuit_429_threshold = circuit_429_threshold
        self.circuit_pause_s = circuit_pause_s
        self.downscale_factor = downscale_factor
        self.upscale_factor = upscale_factor
        self.upscale_check_s = upscale_check_s

        self.consecutive_429 = 0
        self.circuit_open_until = 0.0
        self.last_429_at = 0.0
        self.last_upscale_at = time.time()

        self.lock = threading.Lock()

        # 统计
        self.total_acquires = 0
        self.total_429 = 0
        self.total_success = 0

    def acquire(self) -> None:
        """阻塞直到能发 1 个请求 (拿到 token + circuit 关).

        - circuit 开 (熔断中) → 等到 circuit_open_until
        - token bucket 空 → 等 refill
        """
        while True:
            with self.lock:
                now = time.time()
                # 1. circuit breaker 检查
                if now < self.circuit_open_until:
                    wait = self.circuit_open_until - now
                    self._dbg(f"[throttle] circuit open, wait {wait:.1f}s")
                else:
                    # 2. refill tokens
                    elapsed = now - self.last_refill
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                    self.last_refill = now

                    # 3. try acquire
                    if self.tokens >= 1:
                        self.tokens -= 1
                        self.total_acquires += 1
                        return
                    # token 不够, 等 refill
                    wait = (1 - self.tokens) / self.refill_rate

            time.sleep(min(wait, 1.0))

    def on_success(self) -> None:
        """200 OK: 重置 consecutive 429 + 考虑升 RPS."""
        with self.lock:
            self.consecutive_429 = 0
            self.total_success += 1
            now = time.time()
            # 长时间无 429 → 自适应升 RPS
            if (now - self.last_429_at >= self.upscale_check_s and
                    now - self.last_upscale_at >= self.upscale_check_s and
                    self.refill_rate < self.max_rps):
                new_rps = min(self.max_rps, self.refill_rate * self.upscale_factor)
                self._dbg(f"[throttle] no 429 in {self.upscale_check_s}s, RPS up: {self.refill_rate:.2f} → {new_rps:.2f}")
                self.refill_rate = new_rps
                self.capacity = new_rps
                self.last_upscale_at = now

    def on_429(self, retry_after: Optional[float] = None, body: str = "") -> float:
        """收到 429: 触发 backoff + 自适应降速.

        Returns: 调用方应等待的秒数 (before retry).
        Raises: AccountExhaustedError if body 含 quota/credit exhausted signal.
        """
        with self.lock:
            # 区分 a/b 两种限流
            if any(kw in body.lower() for kw in ("quota exceeded", "credit", "insufficient", "exhausted", "余额")):
                raise AccountExhaustedError(f"额度耗尽 (非速率限): {body[:200]}")

            self.consecutive_429 += 1
            self.total_429 += 1
            self.last_429_at = time.time()

            # backoff: retry_after 优先, 否则 2^N 渐进
            if retry_after is not None and retry_after > 0:
                wait = min(retry_after, self.backoff_max_s)
            else:
                wait = min(self.backoff_base ** self.consecutive_429, self.backoff_max_s)

            # 自适应降 RPS
            new_rps = max(self.min_rps, self.refill_rate * self.downscale_factor)
            self._dbg(f"[throttle] 429 #{self.consecutive_429}, wait {wait:.1f}s, RPS: {self.refill_rate:.2f} → {new_rps:.2f}")
            self.refill_rate = new_rps
            self.capacity = new_rps

            # 连续 N 个 429 → 熔断
            if self.consecutive_429 >= self.circuit_429_threshold:
                pause = self.circuit_pause_s
                self.circuit_open_until = time.time() + pause
                self._dbg(f"[throttle] 🔴 CIRCUIT OPEN: 连续 {self.consecutive_429} 个 429, 熔断 {pause}s")

            return wait

    def stats(self) -> dict:
        """当前限速状态摘要."""
        with self.lock:
            return {
                "current_rps": self.refill_rate,
                "tokens_now": self.tokens,
                "total_acquires": self.total_acquires,
                "total_success": self.total_success,
                "total_429": self.total_429,
                "rate_429": self.total_429 / max(1, self.total_acquires),
                "consecutive_429": self.consecutive_429,
                "circuit_open": time.time() < self.circuit_open_until,
                "circuit_remaining_s": max(0, self.circuit_open_until - time.time()),
            }

    @staticmethod
    def _dbg(msg: str) -> None:
        # 简单打印, 后续可换成 logging
        print(msg, flush=True)


# === 余额查询 (区分速率 vs 额度) ============================================
def query_balance(api_key: str, timeout: float = 5.0) -> Optional[float]:
    """查 gateway dashboard 余额. 返回 RMB float, 失败返 None.

    撞 429 后调用判断:余额够 → 速率限继续 backoff; 余额近 0 → AccountExhaustedError.
    """
    if not os.getenv("LLM_BALANCE_URL"):
        return None
    req = urllib.request.Request(
        os.environ["LLM_BALANCE_URL"],
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=b"",
    )
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        d = json.loads(r.read())
        amt = d.get("grants", {}).get("available_amount")
        return float(amt) if amt is not None else None
    except Exception:
        return None


# === 全局单例 ===============================================================
# 初始 RPS 应该用 benchmark_sustained.py 实测的 "safe RPS" 调
# 默认 5 是保守起点 — 第一次跑会自动升到 max 找上限
GLOBAL_THROTTLE = AdaptiveThrottle(initial_rps=5.0, min_rps=1.0, max_rps=50.0)


if __name__ == "__main__":
    # 单测: 模拟一波 429 看 throttle 行为
    t = AdaptiveThrottle(initial_rps=10)
    print("初始:", t.stats())
    for i in range(5):
        t.acquire()
        wait = t.on_429(body="nginx 429")
        print(f"  第 {i+1} 个 429, wait={wait:.1f}s, stats={t.stats()}")
        time.sleep(0.1)
