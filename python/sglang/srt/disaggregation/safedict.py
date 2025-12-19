#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
-------------------------------------------------
   File Name:     sglang/python/sglang/srt/disaggregation/safedict.py
   Description:   safedict
   Author:        wuzhensheng01@baidu.com
   date:          2025/09/12 done
   project:       None
-------------------------------------------------
"""

import threading
from pydantic import BaseModel
import typing


class ReadWriteLock:
    """线程读写锁：多读共享，写独占"""

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    # 基本锁操作
    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()

    # 上下文管理器
    class _ReadCtx:
        def __init__(self, rwlock):
            self._rwlock = rwlock

        def __enter__(self):
            self._rwlock.acquire_read()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._rwlock.release_read()

    class _WriteCtx:
        def __init__(self, rwlock):
            self._rwlock = rwlock

        def __enter__(self):
            self._rwlock.acquire_write()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._rwlock.release_write()

    def read_ctx(self):
        return ReadWriteLock._ReadCtx(self)

    def write_ctx(self):
        return ReadWriteLock._WriteCtx(self)


class AutoSafeDict:
    """线程安全字典，自动上下文管理优化版"""

    def __init__(self, initial=None):
        self._data = dict(initial) if initial else {}
        self._lock = ReadWriteLock()
        self._local = threading.local()  # 用于线程上下文锁标记

    # ------------------ 内部方法 ------------------
    def _in_read_context(self):
        return getattr(self._local, 'read_ctx', False)

    def _in_write_context(self):
        return getattr(self._local, 'write_ctx', False)

    def _enter_read(self):
        if not self._in_read_context() and not self._in_write_context():
            self._lock.acquire_read()
            self._local.read_ctx = True

    def _exit_read(self):
        if getattr(self._local, 'read_ctx', False):
            self._lock.release_read()
            self._local.read_ctx = False

    def _enter_write(self):
        if not self._in_write_context():
            self._lock.acquire_write()
            self._local.write_ctx = True

    def _exit_write(self):
        if getattr(self._local, 'write_ctx', False):
            self._lock.release_write()
            self._local.write_ctx = False

    # ------------------ 上下文管理 ------------------
    class read_ctx:
        def __init__(self, outer):
            self._outer = outer

        def __enter__(self):
            self._outer._enter_read()
            return self._outer

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._outer._exit_read()

    class write_ctx:
        def __init__(self, outer):
            self._outer = outer

        def __enter__(self):
            self._outer._enter_write()
            return self._outer

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._outer._exit_write()

    def read(self):
        return AutoSafeDict.read_ctx(self)

    def write(self):
        return AutoSafeDict.write_ctx(self)

    # ------------------ 原生接口 ------------------
    def __getitem__(self, key):
        self._enter_read()
        try:
            return self._data[key]
        finally:
            self._exit_read()

    def __setitem__(self, key, value):
        self._enter_write()
        try:
            self._data[key] = value
        finally:
            self._exit_write()

    def __delitem__(self, key):
        self._enter_write()
        try:
            del self._data[key]
        finally:
            self._exit_write()

    def __contains__(self, key):
        self._enter_read()
        try:
            return key in self._data
        finally:
            self._exit_read()

    def __len__(self):
        self._enter_read()
        try:
            return len(self._data)
        finally:
            self._exit_read()

    def get(self, key, default=None):
        self._enter_read()
        try:
            return self._data.get(key, default)
        finally:
            self._exit_read()

    def keys(self):
        self._enter_read()
        try:
            return list(self._data.keys())
        finally:
            self._exit_read()

    def values(self):
        self._enter_read()
        try:
            return list(self._data.values())
        finally:
            self._exit_read()

    def items(self):
        self._enter_read()
        try:
            return list(self._data.items())
        finally:
            self._exit_read()

    def update(self, other):
        self._enter_write()
        try:
            self._data.update(other)
        finally:
            self._exit_write()

    def clear(self):
        self._enter_write()
        try:
            self._data.clear()
        finally:
            self._exit_write()


class PodLoadInfo(BaseModel):
    session_id: typing.Optional[str]
    type_pod: typing.Optional[str]  # P or D
    unbootstrapped_req: typing.Optional[int] = None  # 处理握手队列队列的长度
    waiting_queue: typing.Optional[int]= None  # 等待队列的长度
    transferring_req: typing.Optional[int] = None  # 处于KvCache传输的队列长度
    input_throughput: typing.Optional[float] = None  # prefill当前的吞吐
    cached_token: typing.Optional[int] = None  # 有缓存的token
    new_token: typing.Optional[int] = None  # 新产生的token

    num_running_reqs: typing.Optional[int] = None  # Decode阶段当前的推理Batch
    pre_allocated_usage: typing.Optional[str] = None  # disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens
    retracted_req: typing.Optional[int] = None
    cuda_graph: typing.Optional[bool] = None
    gen_throughput: typing.Optional[float] = None



if __name__ == '__main__':
    import time

    a = AutoSafeDict()


    def writer(name):
        with a.write():  # 自动锁住写锁
            for i in range(3):
                a[f"{name}-{i}"] = i
                print(f"{name} wrote {name}-{i}")
                time.sleep(0.1)


    def reader(name):
        with a.read():  # 自动锁住读锁
            for _ in range(3):
                print(f"{name} reads {list(a.items())}")
                time.sleep(0.15)


    threads = [
        threading.Thread(target=writer, args=("W1",)),
        threading.Thread(target=writer, args=("W2",)),
        threading.Thread(target=reader, args=("R1",)),
        threading.Thread(target=reader, args=("R2",)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()
