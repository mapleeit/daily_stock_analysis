# -*- coding: utf-8 -*-
"""
DataFetcherManager 港股路由与实时行情去重测试
"""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _install_provider_stubs():
    """
    安装 data_provider 子模块 stub，避免测试环境缺少第三方依赖时导入失败。
    """
    provider_modules = {
        'data_provider.efinance_fetcher': {'EfinanceFetcher': type('EfinanceFetcher', (), {})},
        'data_provider.tushare_fetcher': {'TushareFetcher': type('TushareFetcher', (), {})},
        'data_provider.pytdx_fetcher': {'PytdxFetcher': type('PytdxFetcher', (), {})},
        'data_provider.baostock_fetcher': {'BaostockFetcher': type('BaostockFetcher', (), {})},
        'data_provider.yfinance_fetcher': {'YfinanceFetcher': type('YfinanceFetcher', (), {})},
    }

    for module_name, attrs in provider_modules.items():
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[module_name] = module


_install_provider_stubs()


class _StubDailyFetcher:
    """最小化日线 fetcher stub，仅用于路由测试。"""

    def __init__(self, name, priority, calls, should_succeed=False):
        self.name = name
        self.priority = priority
        self._calls = calls
        self._should_succeed = should_succeed

    def get_daily_data(self, stock_code, start_date=None, end_date=None, days=30):
        self._calls.append(self.name)
        if self._should_succeed:
            return pd.DataFrame({'close': [1.0]})
        raise RuntimeError(f"{self.name} failed")


class _StubRealtimeFetcher:
    """最小化实时行情 fetcher stub，用于港股 source 去重测试。"""

    def __init__(self):
        self.name = "AkshareFetcher"
        self.priority = 1
        self.calls = []

    def get_realtime_quote(self, stock_code, source="em"):
        self.calls.append((stock_code, source))
        return None


class TestHKDailyRouting(unittest.TestCase):
    def test_hk_daily_route_skips_non_hk_fetchers(self):
        """港股日线应仅尝试 Akshare/Yfinance，不应尝试 efinance/pytdx。"""
        from data_provider.base import DataFetcherManager

        calls = []
        fetchers = [
            _StubDailyFetcher("EfinanceFetcher", 0, calls, should_succeed=False),
            _StubDailyFetcher("AkshareFetcher", 1, calls, should_succeed=True),
            _StubDailyFetcher("PytdxFetcher", 2, calls, should_succeed=False),
            _StubDailyFetcher("BaostockFetcher", 3, calls, should_succeed=False),
            _StubDailyFetcher("YfinanceFetcher", 4, calls, should_succeed=True),
        ]
        manager = DataFetcherManager(fetchers=fetchers)

        df, source = manager.get_daily_data("HK00700")

        self.assertFalse(df.empty)
        self.assertEqual(source, "AkshareFetcher")
        self.assertEqual(calls, ["AkshareFetcher"])

    def test_hk_daily_route_fallback_to_yfinance_only(self):
        """港股 Akshare 失败后应直接回退 Yfinance，不经过 A 股专用源。"""
        from data_provider.base import DataFetcherManager

        calls = []
        fetchers = [
            _StubDailyFetcher("EfinanceFetcher", 0, calls, should_succeed=False),
            _StubDailyFetcher("AkshareFetcher", 1, calls, should_succeed=False),
            _StubDailyFetcher("TushareFetcher", 2, calls, should_succeed=False),
            _StubDailyFetcher("PytdxFetcher", 2, calls, should_succeed=False),
            _StubDailyFetcher("BaostockFetcher", 3, calls, should_succeed=False),
            _StubDailyFetcher("YfinanceFetcher", 4, calls, should_succeed=True),
        ]
        manager = DataFetcherManager(fetchers=fetchers)

        df, source = manager.get_daily_data("HK03690")

        self.assertFalse(df.empty)
        self.assertEqual(source, "YfinanceFetcher")
        self.assertEqual(calls, ["AkshareFetcher", "YfinanceFetcher"])

    def test_daily_failure_negative_cache_skips_repeated_failed_fetcher(self):
        """
        同一 manager 实例内，某日线数据源刚失败后应进入短期负缓存并被跳过。
        """
        from data_provider.base import DataFetcherManager

        calls = []
        fetchers = [
            _StubDailyFetcher("AkshareFetcher", 1, calls, should_succeed=False),
            _StubDailyFetcher("YfinanceFetcher", 2, calls, should_succeed=True),
        ]
        manager = DataFetcherManager(fetchers=fetchers)
        manager._daily_failure_ttl = 3600.0  # 放大窗口，确保第二次调用命中负缓存

        # 第一次：Akshare 失败，Yfinance 成功
        df1, source1 = manager.get_daily_data("HK00700")
        # 第二次：Akshare 应被负缓存跳过，直接命中 Yfinance
        df2, source2 = manager.get_daily_data("HK03690")

        self.assertFalse(df1.empty)
        self.assertFalse(df2.empty)
        self.assertEqual(source1, "YfinanceFetcher")
        self.assertEqual(source2, "YfinanceFetcher")
        # 如果负缓存生效，Akshare 仅应被调用一次（第一次）
        self.assertEqual(calls, ["AkshareFetcher", "YfinanceFetcher", "YfinanceFetcher"])


class TestHKRealtimeDedup(unittest.TestCase):
    def test_hk_realtime_akshare_source_deduplicated(self):
        """
        港股实时行情在 tencent/akshare_sina/akshare_em 优先级下，
        应只调用一次 AkshareFetcher（避免同一 HK 接口重复请求）。
        """
        from data_provider.base import DataFetcherManager

        import src.config as app_config

        realtime_fetcher = _StubRealtimeFetcher()
        manager = DataFetcherManager(fetchers=[realtime_fetcher])

        with patch.object(app_config, 'get_config', return_value=SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,akshare_em",
        )):
            quote = manager.get_realtime_quote("HK09988")

        self.assertIsNone(quote)
        self.assertEqual(len(realtime_fetcher.calls), 1)
        self.assertEqual(realtime_fetcher.calls[0][0], "HK09988")


class _DummyBreaker:
    def __init__(self):
        self.failure_count = 0
        self.success_count = 0

    def is_available(self, _source_key):
        return True

    def record_success(self, _source_key):
        self.success_count += 1

    def record_failure(self, _source_key, _error=None):
        self.failure_count += 1


class TestAkshareHKNegativeCache(unittest.TestCase):
    def test_hk_realtime_failure_negative_cache(self):
        """
        首次港股实时请求失败后，负缓存窗口内再次请求应直接跳过 API 调用。
        """
        from data_provider import akshare_fetcher as akf

        # 重置模块级缓存，避免测试间相互影响
        akf._hk_realtime_cache.update({
            'data': None,
            'timestamp': 0,
            'ttl': 300,
            'failure_timestamp': 0,
            'failure_ttl': 180,
        })

        call_counter = {'n': 0}
        fake_akshare = types.ModuleType('akshare')

        def _fail_hk_spot():
            call_counter['n'] += 1
            raise RuntimeError("mock hk api failure")

        fake_akshare.stock_hk_spot_em = _fail_hk_spot
        breaker = _DummyBreaker()

        with patch.object(akf, 'get_config', return_value=SimpleNamespace(enable_eastmoney_patch=False)), \
             patch.object(akf, 'get_realtime_circuit_breaker', return_value=breaker), \
             patch.dict(sys.modules, {'akshare': fake_akshare}):
            fetcher = akf.AkshareFetcher()
            # 避免测试中的真实 sleep
            fetcher._set_random_user_agent = lambda: None
            fetcher._enforce_rate_limit = lambda: None

            first = fetcher._get_hk_realtime_quote("HK00700")
            second = fetcher._get_hk_realtime_quote("HK00700")

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(call_counter['n'], 1)


if __name__ == '__main__':
    unittest.main()
