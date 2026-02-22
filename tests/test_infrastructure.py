"""Tests for infrastructure components: version, connection pools, rate limiting."""

from __future__ import annotations

import pytest

import grip
from grip.api.rate_limit import SlidingWindowRateLimiter
from grip.pool import get_http_pool, get_provider_pool, shutdown_pools

# ---------------------------------------------------------------------------
# Version consistency
# ---------------------------------------------------------------------------


class TestVersionConsistency:
    def test_version_is_0_2_0(self):
        assert grip.__version__ == "0.2.0"

    def test_version_is_string(self):
        assert isinstance(grip.__version__, str)

    def test_version_has_three_parts(self):
        parts = grip.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# Connection pool lifecycle
# ---------------------------------------------------------------------------


class TestPoolLifecycle:
    def test_get_http_pool_returns_same_instance(self):
        pool1 = get_http_pool()
        pool2 = get_http_pool()
        assert pool1 is pool2

    def test_get_provider_pool_returns_same_instance(self):
        pool1 = get_provider_pool()
        pool2 = get_provider_pool()
        assert pool1 is pool2

    @pytest.mark.asyncio
    async def test_shutdown_pools_resets_globals(self):
        import grip.pool as pool_module

        get_http_pool()
        get_provider_pool()
        assert pool_module._global_http_pool is not None
        assert pool_module._global_provider_pool is not None

        await shutdown_pools()
        assert pool_module._global_http_pool is None
        assert pool_module._global_provider_pool is None


# ---------------------------------------------------------------------------
# API rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, remaining, retry_after = limiter.is_allowed("client1")
            assert allowed is True
            assert retry_after == 0.0

    def test_blocks_over_limit(self):
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.is_allowed("client1")

        allowed, remaining, retry_after = limiter.is_allowed("client1")
        assert allowed is False
        assert remaining == 0
        assert retry_after > 0

    def test_independent_keys(self):
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("client_a")
        limiter.is_allowed("client_a")

        allowed, _, _ = limiter.is_allowed("client_b")
        assert allowed is True

    def test_remaining_decreases(self):
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)

        _, remaining1, _ = limiter.is_allowed("x")
        _, remaining2, _ = limiter.is_allowed("x")

        assert remaining1 == 4
        assert remaining2 == 3

    def test_cleanup_removes_stale(self):
        limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=0)
        limiter.is_allowed("stale_client")
        removed = limiter.cleanup()
        assert removed >= 1
