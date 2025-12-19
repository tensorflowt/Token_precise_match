"""
created by yansiyu01@baidu.com at 2025/07/31
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import os
import queue
import socket
import time
import threading
import requests
import zmq
import ctypes
import struct
import numpy as np
import numpy.typing as npt

from collections import defaultdict
from typing import Dict, List, Optional, Union, Tuple, Set

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.p2p.transfer_engine import P2PTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    format_tcp_address,
    get_int_env_var,
    is_valid_ipv6_address,
)

logger = logging.getLogger(__name__)


class P2PTransferError(Exception):
    """
    P2PTransferError 异常类，表示 P2P 传输过程中发生的错误
    """
    def __init__(self, bootstrap_room: int, failure_reason: str):
        """
        初始化
        """
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        """
        返回 P2PTransferError 对象的字符串表示形式
        """
        return f"P2PTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    """
    KVCache 数据的元信息
    """
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]
    state_indices: Optional[List[int]]


# decode
@dataclasses.dataclass
class TransferInfo:
    """
    Decode 接收 KVCache 所需的目标传输信息
    """
    room: int
    endpoint: str
    dst_port: int
    p2p_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: List[int]
    required_dst_info_num: int
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        """
        从 ZMQ 消息中解析并创建一个新的实例
        """
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            if msg[6] == b"":
                dst_state_indices = []
            else:
                dst_state_indices = list(np.frombuffer(msg[6], dtype=np.int32))
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            p2p_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    """
    KVCache 注册信息
    """
    room: str
    endpoint: str
    dst_port: int
    p2p_session_id: str
    dst_kv_ptrs: list[int] 
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: list[int]
    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        """
        从 ZMQ 消息中解析并创建一个新的实例
        """
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            p2p_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
        )


class AuxDataCodec:
    """
    AuxDataCodec 序列化和反序列化辅助数据缓冲区的工具类
    """

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """
        从指定的内存地址读取指定长度的数据，并将其转换为字节串
        """
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """
        将指定的数据写入到指定的内存地址中
        """
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


class P2PKVManager(CommonKVManager):
    """
    P2PKVManager 类
    """
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        """
        初始化 P2PKVManager 
        """
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.server_args = server_args 
        self.state_handles = []  
        self.engine = P2PTransferEngine(self.local_ip, self.kv_args.gpu_id)
        self.decode_physical_gpu_ids = {}
        self.register_buffer_to_engine()
        
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.start_prefill_thread()
            self.session_failures = defaultdict(int) 
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count() 
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE", 
                min(max(4, int(0.75 * cpu_count) // 8), 12), 
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4) 
            self.transfer_queues: List[FastQueue] = [ 
                FastQueue() for _ in range(transfer_queue_size) 
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be " 
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}." 
            )
            self.executors = [ 
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size 
                )
                for _ in range(transfer_queue_size)
            ]
            for queue, executor in zip(self.transfer_queues, self.executors): 
                threading.Thread(
                    target=self.transfer_worker, args=(queue, executor), daemon=True
                ).start()

            self.bootstrap_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 30 
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set) 
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
        
            self.heartbeat_interval = max( 
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0 
            )
            self.max_failures = max( 
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_decode_thread()
            self.waiting_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300
            )
        
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()
        self.handle_size = 64
        self.p2p_batch_limit = int(os.getenv("SGLANG_P2P_BATCH_LIMIT", "512") or "512")
        self.transfer_timeout = float(os.getenv("SGLANG_P2P_TRANSFER_TIMEOUT", "60") or "60")
        # KVCache 传输统计，以 room (每个请求) 为 key，保存 { "kv_bytes": int, "kv_time_ms": float, "kv_tokens": int }
        self.kv_transfer_stats: Dict[int, Dict[str, float | int]] = {}
        self.kv_transfer_stats_lock = threading.Lock()
        self.enable_kvcache_log = get_int_env_var("SGLANG_KVCACHE_LOG", 0) 

    def register_buffer_to_engine(self):
        """
        将KV缓冲区注册到引擎中
        """
        logger.info("Start registering KV buffers to P2P engines.")
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            # KV handles
            self.kv_handles = []
            for ptr in self.kv_args.kv_data_ptrs:
                kv_handle = self.engine.register_buffer(ptr)
                self.kv_handles.append(kv_handle)

            # State/extra pool handles
            self.state_handles = []
            if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
                for ptr in getattr(self.kv_args, "state_data_ptrs", []) or []:
                    state_handle = self.engine.register_buffer(ptr)
                    self.state_handles.append(state_handle)

    def start_prefill_thread(self):
        """
        启动预填充线程
        """
        self._bind_server_socket()

        def bootstrap_thread():
            """
            预填充线程函数
            """
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                p2p_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    physical_gpu_id = int(waiting_req_bytes[6].decode("ascii"))
                    self.decode_physical_gpu_ids[p2p_session_id] = physical_gpu_id
                    
                    # KV handles
                    handles_bytes = waiting_req_bytes[4]
                    if len(handles_bytes) % self.handle_size != 0:
                        logger.error(f"Invalid handles_bytes length={len(handles_bytes)} (handle_size={self.handle_size})")
                        continue
                    num_kv_handles = len(handles_bytes) // self.handle_size

                    dst_kv_ptrs = [
                        handles_bytes[i * self.handle_size:(i + 1) * self.handle_size]
                        for i in range(num_kv_handles)
                    ]

                    for kv_handle in dst_kv_ptrs:
                        try:
                            result = self.engine.register_d_handle(kv_handle)
                        except Exception as e:
                            logger.exception(f"register_d_handle failed: {e}")

                    # AUX raw ptrs
                    aux_ptrs = list(struct.unpack(f"{len(waiting_req_bytes[5]) // 8}Q", waiting_req_bytes[5]))
                    
                    # State handles
                    state_handles_bytes = waiting_req_bytes[7] if len(waiting_req_bytes) > 7 else b""
                    dst_state_ptrs: List[Union[int, bytes]] = []
                    if state_handles_bytes:
                        if len(state_handles_bytes) % self.handle_size != 0:
                            logger.error(f"Invalid state_handles_bytes length={len(state_handles_bytes)} (handle_size={self.handle_size})")
                        else:
                            num_state_handles = len(state_handles_bytes) // self.handle_size
                            dst_state_ptrs = [
                                state_handles_bytes[i * self.handle_size:(i + 1) * self.handle_size]
                                for i in range(num_state_handles)
                            ]
                            for h in dst_state_ptrs:
                                try:
                                    self.engine.register_d_handle(h)
                                except Exception as e:
                                    logger.exception(f"register_d_handle(State) failed: {e}")
                    # 用于 TP slicing
                    dst_tp_rank = int(waiting_req_bytes[8].decode("ascii")) if len(waiting_req_bytes) > 8 else 0
                    dst_attn_tp_size = int(waiting_req_bytes[9].decode("ascii")) if len(waiting_req_bytes) > 9 else 0
                    dst_kv_item_len = int(waiting_req_bytes[10].decode("ascii")) if len(waiting_req_bytes) > 10 else 0

                    self.decode_kv_args_table[p2p_session_id] = KVArgsRegisterInfo(
                        room=room,
                        endpoint=waiting_req_bytes[1].decode("ascii"),
                        dst_port=int(waiting_req_bytes[2].decode("ascii")),
                        p2p_session_id=p2p_session_id,
                        dst_kv_ptrs=dst_kv_ptrs,
                        dst_aux_ptrs=aux_ptrs,
                        dst_state_data_ptrs=dst_state_ptrs,
                        dst_tp_rank=dst_tp_rank,
                        dst_attn_tp_size=dst_attn_tp_size,
                        dst_kv_item_len=dst_kv_item_len,
                    )
                else:
                    # 传输请求
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(waiting_req_bytes[0].decode("ascii"))
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}
                    
                    self.transfer_infos[room][p2p_session_id] = TransferInfo.from_zmq(waiting_req_bytes)
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)
        
        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def send_kvcache(
        self,
        req: TransferInfo, 
        p2p_session_id: str,  
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: List[bytes],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        发送 KVCache
        更新：使用 engine.transfer_many 批量发送 KVCache（SGLANG_P2P_BATCH_LIMIT设置batch大小）
        """
        try:
            dst_physical_gpu_id = self.decode_physical_gpu_ids.get(p2p_session_id)
        
            if dst_physical_gpu_id is None:
                logger.error(f"Physical GPU ID not found for session {p2p_session_id}")
                return 1

            src_physical_gpu_id = self.kv_args.gpu_id # 源 GPU
            
            prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
                prefill_kv_indices, dst_kv_indices
            )

            batch_src_ptrs: list[int] = []
            batch_src_devs: list[int] = []
            batch_dst_handles: list[bytes] = []
            batch_dst_devs: list[int] = []
            batch_offsets: list[int] = []
            batch_lengths: list[int] = []

            total_tokens = 0
            start_time = time.perf_counter()

            num_layers = len(self.kv_args.kv_data_ptrs)
            for layer_id in range(num_layers):
                base_ptr = int(self.kv_args.kv_data_ptrs[layer_id])
                item_len = int(self.kv_args.kv_item_lens[layer_id])  # 一个block的字节数
                dst_handle = dst_kv_ptrs[layer_id]

                for prefill_block, dst_block in zip(prefill_kv_blocks, dst_kv_blocks):
                    if not prefill_block or not dst_block:
                        continue
                    first_src_block = int(prefill_block[0])
                    first_dst_block = int(dst_block[0])
                    length_bytes = int(len(prefill_block) * item_len)

                    src_ptr = base_ptr + first_src_block * item_len
                    dst_off = first_dst_block * item_len

                    batch_src_ptrs.append(src_ptr)
                    batch_src_devs.append(int(src_physical_gpu_id))
                    batch_dst_handles.append(dst_handle)
                    batch_dst_devs.append(int(dst_physical_gpu_id))
                    batch_offsets.append(dst_off)
                    batch_lengths.append(length_bytes)

                    total_tokens += len(prefill_block)

            if not batch_src_ptrs:
                logger.debug("No KV transfer tasks to send.")
                return 0

            # batch 提交（transfer_many），返回一个句柄
            handles = []
            total_bytes = 0
            batch_limit = self.p2p_batch_limit
            timeout_s = self.transfer_timeout
            for i in range(0, len(batch_src_ptrs), batch_limit):
                j = i + batch_limit
                sub_src_ptrs = batch_src_ptrs[i:j]
                sub_src_devs = batch_src_devs[i:j]
                sub_dst_handles = batch_dst_handles[i:j]
                sub_dst_devs = batch_dst_devs[i:j]
                sub_offsets = batch_offsets[i:j]
                sub_lengths = batch_lengths[i:j]

                if hasattr(self.engine, "transfer_many"):
                    h = self.engine.transfer_many(
                        src_ptrs=sub_src_ptrs,
                        src_devs=sub_src_devs,
                        dst_handles=sub_dst_handles,
                        dst_devs=sub_dst_devs,
                        dst_offsets=sub_offsets,
                        lengths=sub_lengths,
                    )
                    handles.append(h)
                else:
                    sub_handles = []
                    for sp, sd, dh, dd, off, ln in zip(
                        sub_src_ptrs, sub_src_devs, sub_dst_handles, sub_dst_devs, sub_offsets, sub_lengths
                    ):
                        sub_handles.append(self.engine.transfer(sp, sd, dh, dd, off, ln))

                    class _BatchHandle:
                        def __init__(self, hs): self._hs = hs
                        def is_done(self): return all(h.is_done() for h in self._hs)
                        def wait(self):
                            for h in self._hs:
                                if hasattr(h, "wait"): h.wait()
                                else:
                                    while not h.is_done(): time.sleep(0.001)
                    handles.append(_BatchHandle(sub_handles))

                total_bytes += sum(sub_lengths)

            # 统一等待所有batch完成（轮询）
            deadline = time.perf_counter() + timeout_s
            while True:
                remaining = [h for h in handles if not h.is_done()]
                if not remaining:
                    break
                if time.perf_counter() > deadline:
                    logger.error(f"P2P transfer_many timeout after {timeout_s}s")
                    return 1
                time.sleep(0.002)

            # 统计信息
            total_time_ms = (time.perf_counter() - start_time) * 1000.0
            with self.kv_transfer_stats_lock:
                self.kv_transfer_stats[req.room] = {
                    "kv_bytes": int(total_bytes),
                    "kv_time_ms": float(total_time_ms),
                    "kv_tokens": int(total_tokens),
                }

            return 0

        except Exception as e:
            logger.exception(f"P2P KV cache transfer failed: {e}")
            return 1

    def send_kvcache_slice(
        self,
        req: TransferInfo,
        p2p_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_handles: List[bytes],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        非 MLA 且 TP 不一致时：逐 token-slot、按 head-slice 发送 KV cache
        """
        try:
            dst_physical_gpu_id = self.decode_physical_gpu_ids.get(p2p_session_id)
            if dst_physical_gpu_id is None:
                logger.error(f"Physical GPU ID not found for session {p2p_session_id}")
                return 1
            src_physical_gpu_id = int(self.kv_args.gpu_id)

            page_size = int(self.kv_args.page_size)
            num_kv_heads = int(self.kv_args.kv_head_num)
            local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
            dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

            src_item_len = int(self.kv_args.kv_item_lens[0])
            bytes_per_token_src = src_item_len // page_size
            bytes_per_token_dst = dst_kv_item_len // page_size

            src_heads_per_rank = num_kv_heads
            dst_heads_per_rank = num_kv_heads * self.attn_tp_size // dst_attn_tp_size
            bytes_per_head_slice = dst_kv_item_len // page_size // dst_heads_per_rank

            if self.attn_tp_size > dst_attn_tp_size:
                src_head_start = 0
                num_heads_to_send = src_heads_per_rank
                dst_head_start = local_tp_rank_in_group * src_heads_per_rank
            else:
                src_head_start = (dst_tp_rank_in_group * dst_heads_per_rank) % src_heads_per_rank
                num_heads_to_send = dst_heads_per_rank
                dst_head_start = 0

            src_head_slice_offset = src_head_start * bytes_per_head_slice
            dst_head_slice_offset = dst_head_start * bytes_per_head_slice
            heads_bytes_per_token = num_heads_to_send * bytes_per_head_slice

            if heads_bytes_per_token > bytes_per_token_dst:
                logger.error(f"[{p2p_session_id}] slice size {heads_bytes_per_token} exceeds dst token size {bytes_per_token_dst}")
                return 1

            batch_src_ptrs: List[int] = []
            batch_src_devs: List[int] = []
            batch_dst_handles: List[bytes] = []
            batch_dst_devs: List[int] = []
            batch_offsets: List[int] = []
            batch_lengths: List[int] = []

            for layer_id in range(len(self.kv_args.kv_data_ptrs)):
                base_ptr = int(self.kv_args.kv_data_ptrs[layer_id])
                dst_handle = dst_kv_handles[layer_id]

                for i in range(len(prefill_kv_indices)):
                    prefill_page_idx = int(prefill_kv_indices[i])
                    decode_page_idx = int(dst_kv_indices[i])

                    src_page_start = base_ptr + prefill_page_idx * src_item_len
                    dst_page_off = decode_page_idx * dst_kv_item_len

                    for token_slot in range(page_size):
                        src_token_slot_addr = src_page_start + token_slot * bytes_per_token_src + src_head_slice_offset
                        dst_token_slot_off = dst_page_off + token_slot * bytes_per_token_dst + dst_head_slice_offset

                        batch_src_ptrs.append(src_token_slot_addr)
                        batch_src_devs.append(src_physical_gpu_id)
                        batch_dst_handles.append(dst_handle)
                        batch_dst_devs.append(dst_physical_gpu_id)
                        batch_offsets.append(dst_token_slot_off)
                        batch_lengths.append(heads_bytes_per_token)

            if not batch_src_ptrs:
                return 0

            batch_limit = self.p2p_batch_limit
            timeout_s = self.transfer_timeout

            handles = []
            for i in range(0, len(batch_src_ptrs), batch_limit):
                j = i + batch_limit
                h = self.engine.transfer_many(
                    src_ptrs=batch_src_ptrs[i:j],
                    src_devs=batch_src_devs[i:j],
                    dst_handles=batch_dst_handles[i:j],
                    dst_devs=batch_dst_devs[i:j],
                    dst_offsets=batch_offsets[i:j],
                    lengths=batch_lengths[i:j],
                )
                handles.append(h)

            deadline = time.perf_counter() + timeout_s
            while True:
                if all(h.is_done() for h in handles):
                    break
                if time.perf_counter() > deadline:
                    logger.error(f"P2P send_kvcache_slice timeout after {timeout_s}s")
                    return 1
                time.sleep(0.002)

            return 0

        except Exception as e:
            logger.exception(f"P2P KV slice transfer failed: {e}")
            return 1


    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        """
        发送AUX
        """
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                dst_port=req.dst_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )

        return 0

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        """
        发送AUX数据 
        """
        try:
            socket = self._connect(
                format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
            )

            socket.send_multipart(
                [
                    P2PKVManager.AUX_DATA_HEADER,
                    str(room).encode("ascii"),
                    str(buffer_index).encode("ascii"),
                    str(aux_index).encode("ascii"),
                    struct.pack(">I", len(data)),
                    data,
                ]
            )
            return 0
        except Exception as e:
            logger.exception(f"Failed to send aux: {e}")
            return 1

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: List[int],
        dst_state_handles: List[Union[int, bytes]],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        发送 state/extra pool 数据（与 Mooncake 一致）
            1. mamba：单 index 整块拷贝
            2. swa/nsa：与 KV 类似，按 page index 分块拷贝
        """
        state_type = getattr(self.kv_args, "state_type", "none")
        if not prefill_state_indices or not dst_state_handles:
            return 0

        # 非 MLA + TP 不一致时的 extra 传输目前不支持（与 Mooncake 对齐）
        target = self.decode_kv_args_table[req.p2p_session_id]
        if (not self.is_mla_backend) and (self.attn_tp_size != target.dst_attn_tp_size):
            raise RuntimeError("PD Disaggregation does NOT support different TP sizes for non-MLA models' extra/state pools yet.")

        if state_type == "mamba":
            return self._send_mamba_state(req, prefill_state_indices, dst_state_handles)
        elif state_type in ["swa", "nsa"]:
            return self._send_state_generic(req, prefill_state_indices, dst_state_handles)
        else:
            return 0

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_state_indices: List[int],
        dst_state_handles: List[Union[int, bytes]],
    ):
        """
        Mamba：单 index 的整块拷贝（每个层各一块）
        """
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"
        pre_idx = int(prefill_state_indices[0])
        dst_idx = int(req.dst_state_indices[0]) if req.dst_state_indices else 0

        batch_src_ptrs: List[int] = []
        batch_src_devs: List[int] = []
        batch_dst_handles: List[bytes] = []
        batch_dst_devs: List[int] = []
        batch_offsets: List[int] = []
        batch_lengths: List[int] = []

        dst_physical_gpu_id = self.decode_physical_gpu_ids.get(req.p2p_session_id)
        if dst_physical_gpu_id is None:
            logger.error(f"Physical GPU ID not found for session {req.p2p_session_id}")
            return 1

        src_physical_gpu_id = int(self.kv_args.gpu_id)

        for i, dst_handle in enumerate(dst_state_handles):
            if i >= len(self.kv_args.state_data_ptrs):
                break
            base_ptr = int(self.kv_args.state_data_ptrs[i])
            item_len = int(self.kv_args.state_item_lens[i])
            src_ptr = base_ptr + pre_idx * item_len
            dst_off = dst_idx * item_len

            batch_src_ptrs.append(src_ptr)
            batch_src_devs.append(src_physical_gpu_id)
            batch_dst_handles.append(dst_handle)
            batch_dst_devs.append(dst_physical_gpu_id)
            batch_offsets.append(dst_off)
            batch_lengths.append(item_len)

        if not batch_src_ptrs:
            return 0

        batch_limit = self.p2p_batch_limit
        timeout_s = self.transfer_timeout

        handles = []
        for i in range(0, len(batch_src_ptrs), batch_limit):
            j = i + batch_limit
            h = self.engine.transfer_many(
                src_ptrs=batch_src_ptrs[i:j],
                src_devs=batch_src_devs[i:j],
                dst_handles=batch_dst_handles[i:j],
                dst_devs=batch_dst_devs[i:j],
                dst_offsets=batch_offsets[i:j],
                lengths=batch_lengths[i:j],
            )
            handles.append(h)

        deadline = time.perf_counter() + timeout_s
        while True:
            if all(h.is_done() for h in handles):
                break
            if time.perf_counter() > deadline:
                logger.error(f"P2P mamba state transfer timeout after {timeout_s}s")
                return 1
            time.sleep(0.002)

        return 0

    def _send_state_generic(
        self,
        req: TransferInfo,
        prefill_state_indices: List[int],
        dst_state_handles: List[Union[int, bytes]],
    ):
        """
        swa/nsa：与 KV 类似，按 page index 批量拷贝
        """
        pre_idx = np.array(prefill_state_indices, dtype=np.int32)
        dst_idx = np.array(req.dst_state_indices or [], dtype=np.int32)
        if len(pre_idx) == 0 or len(dst_idx) == 0:
            return 0

        pre_blocks, dst_blocks = group_concurrent_contiguous(pre_idx, dst_idx)

        batch_src_ptrs: List[int] = []
        batch_src_devs: List[int] = []
        batch_dst_handles: List[bytes] = []
        batch_dst_devs: List[int] = []
        batch_offsets: List[int] = []
        batch_lengths: List[int] = []

        dst_physical_gpu_id = self.decode_physical_gpu_ids.get(req.p2p_session_id)
        if dst_physical_gpu_id is None:
            logger.error(f"Physical GPU ID not found for session {req.p2p_session_id}")
            return 1
        src_physical_gpu_id = int(self.kv_args.gpu_id)

        num_layers = min(len(self.kv_args.state_data_ptrs), len(dst_state_handles))
        for layer_id in range(num_layers):
            base_ptr = int(self.kv_args.state_data_ptrs[layer_id])
            item_len = int(self.kv_args.state_item_lens[layer_id])
            dst_handle = dst_state_handles[layer_id]

            for s_blk, d_blk in zip(pre_blocks, dst_blocks):
                if not s_blk or not d_blk:
                    continue
                first_s = int(s_blk[0])
                first_d = int(d_blk[0])
                length_bytes = int(len(s_blk) * item_len)

                src_ptr = base_ptr + first_s * item_len
                dst_off = first_d * item_len

                batch_src_ptrs.append(src_ptr)
                batch_src_devs.append(src_physical_gpu_id)
                batch_dst_handles.append(dst_handle)
                batch_dst_devs.append(dst_physical_gpu_id)
                batch_offsets.append(dst_off)
                batch_lengths.append(length_bytes)

        if not batch_src_ptrs:
            return 0

        batch_limit = self.p2p_batch_limit
        timeout_s = self.transfer_timeout

        handles = []
        for i in range(0, len(batch_src_ptrs), batch_limit):
            j = i + batch_limit
            h = self.engine.transfer_many(
                src_ptrs=batch_src_ptrs[i:j],
                src_devs=batch_src_devs[i:j],
                dst_handles=batch_dst_handles[i:j],
                dst_devs=batch_dst_devs[i:j],
                dst_offsets=batch_offsets[i:j],
                lengths=batch_lengths[i:j],
            )
            handles.append(h)

        deadline = time.perf_counter() + timeout_s
        while True:
            if all(h.is_done() for h in handles):
                break
            if time.perf_counter() > deadline:
                logger.error(f"P2P extra-pool transfer timeout after {timeout_s}s")
                return 1
            time.sleep(0.002)

        return 0

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        """
        同步状态
        """
        self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ]
        )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):        
        """
        数据传输
        """
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if req.p2p_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote p2p session {req.p2p_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                                break

                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.p2p_session_id]
                        )
                        if self.is_mla_backend or (
                            self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            ret = self.send_kvcache(
                                req,
                                req.p2p_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,  # KV handles,
                                chunked_dst_kv_indice,
                                executor,
                            )
                        else:
                            ret = self.send_kvcache_slice(
                                req,
                                req.p2p_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,  # KV handles
                                chunked_dst_kv_indice,
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_attn_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )
                        if ret != 0:
                            with self.kv_transfer_stats_lock:
                                self.kv_transfer_stats.pop(kv_chunk.room, None)
                            with self.session_lock:
                                self.session_failures[req.p2p_session_id] += 1
                                if self.session_failures[req.p2p_session_id] >= 1:
                                    self.failed_sessions.add(req.p2p_session_id)
                                    logger.error(
                                        f"Session {req.p2p_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break

                        if kv_chunk.is_last:
                            if kv_chunk.state_indices is not None:
                                target_rank_registration_info: KVArgsRegisterInfo = self.decode_kv_args_table[req.p2p_session_id]
                                ret_extra = self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    target_rank_registration_info.dst_state_data_ptrs,  # state handles
                                    executor,
                                )
                                if ret_extra != 0:
                                    self.record_failure(kv_chunk.room, f"Failed to send extra/state pool for room {kv_chunk.room}")
                                    self.update_status(kv_chunk.room, KVPoll.Failed)
                                    self.sync_status_to_decode_endpoint(req.endpoint, req.dst_port, req.room, KVPoll.Failed, local_rank)
                                    break
                        
                            if self.pp_group.is_last_rank:
                                # Only the last chunk we need to send the aux data
                                ret = self.send_aux(
                                    req,
                                    kv_chunk.prefill_aux_index,
                                    target_rank_registration_info.dst_aux_ptrs,
                                )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status, local_rank
                                    )
                            
                            # 打印单条请求 KVCache 传输统计
                            with self.kv_transfer_stats_lock:
                                stats = self.kv_transfer_stats.pop(kv_chunk.room, None)

                            if self.enable_kvcache_log and stats is not None:
                                kv_bytes = stats.get("kv_bytes", 0)
                                kv_time_ms = stats.get("kv_time_ms", 0.0)
                                kv_tokens = stats.get("kv_tokens", 0)
                                logger.info(
                                    f"[KVCACHE_TRANSFER] room={kv_chunk.room} "
                                    f"bytes={kv_bytes} tokens={kv_tokens} time_ms={kv_time_ms:.2f}"
                                )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )
    
    def _handle_aux_data(self, msg: List[bytes]):
        """
        处理AUX数据 
        """
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}: expected {data_length}, got {len(data)}")
            return
        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )
        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def start_decode_thread(self):
        """
        启动解码线程 
        """
        self._bind_server_socket()

        def decode_thread():
            """
            解码线程 
            """
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == P2PKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                (bootstrap_room, status, prefill_rank) = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = (
                            self.required_prefill_response_num_table[bootstrap_room]
                        )
                        arrived_response_num = len(
                            self.prefill_response_tracker[bootstrap_room]
                        )
                        if arrived_response_num == expected_response_num:
                            self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        f"Failed to get kvcache from prefill instance, it might be dead",
                    )
                self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            """
            心跳检测
            """
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """
        向传输队列中添加传输请求 
        """
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            return

        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
            )
        )

    def check_status(self, bootstrap_room: int):
        """
        检查状态
        """
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        """
        更新状态
        """
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        """
        记录失败原因
        """
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_session_id(self): 
        """
        获取会话ID
        """
        return self.engine.get_session_id() 

    def _handle_node_failure(self, failed_bootstrap_addr):
        """
        处理节点失败的情况 
        """
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_tp_size_table:
                del self.prefill_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests"
        )


class P2PKVSender(CommonKVSender):
    """
    P2PKVSender 类
    """
    def __init__(
        self,
        mgr: P2PKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        """
        初始化函数
        """
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state = None
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """
        向kvmgr发送键值对索引
        """
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room, kv_indices, index_slice, False
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=state_indices,
            )

    def poll(self) -> KVPoll:
        """
        轮询函数
        """
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_timeout:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        """
        清除函数
        """
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)
    
    def failure_exception(self):
        """
        异常处理
        """
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, 
                "P2P transfer failed due to an unknown reason"
            )
        raise P2PTransferError(self.bootstrap_room, failure_reason) 
    
    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class P2PKVReceiver(CommonKVReceiver):
    """
    P2PKVReceiver 类
    """
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: P2PKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        """
        初始化函数
        """      
        self.session_id = mgr.get_session_id()
        self.conclude_state = None
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(
        self, engine_rank, target_dp_group, target_pp_rank
    ):
        """
        从服务器获取引导信息
        """
        try:
            url = (
                f"http://{self.bootstrap_addr}/route"
                f"?engine_rank={engine_rank}"
                f"&target_dp_group={target_dp_group}"
                f"&target_pp_rank={target_pp_rank}"
            )
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _register_kv_args(self):
        """
        注册 KV 参数
        """
        tp_rank = self.kv_mgr.kv_args.engine_rank
        physical_gpu_id = (
            self.kv_mgr.server_args.base_gpu_id +
            tp_rank * self.kv_mgr.server_args.gpu_id_step
        )
                 
        packed_kv_handles = b"".join(self.kv_mgr.kv_handles)
        packed_state_handles = b"".join(self.kv_mgr.state_handles)
        packed_aux_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
        )

        dst_tp_rank = str(tp_rank).encode("ascii")
        dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
        kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]
        dst_kv_item_len = str(kv_item_len).encode("ascii")
        gpu_id_str = str(physical_gpu_id) 

        for bootstrap_info in self.bootstrap_infos:   
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"), 
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_handles,
                        packed_aux_data_ptrs,
                        gpu_id_str.encode("ascii"),  # 发送GPU ID
                        packed_state_handles,
                        dst_tp_rank,
                        dst_attn_tp_size,
                        dst_kv_item_len,
                    ]
                )

    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """
        初始化函数，用于连接到预填充服务器并发送初始化请求
        """
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]

            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        (
                            np.array(
                                state_indices,
                                dtype=np.int32,
                            ).tobytes()
                            if not is_dummy and state_indices is not None
                            else b""
                        ),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        """
        轮询检查状态
        """
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        """
        清除
        """
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:
            self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.prefill_response_tracker:
            self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        """
        失败异常
        """
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "P2P transfer failed due to an unknown reason"
            )
        raise P2PTransferError(self.bootstrap_room, failure_reason)
    
    def abort(self):
        """
        中止
        """
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class P2PKVBootstrapServer(CommonKVBootstrapServer):
    """
    P2PKVBootstrapServer 类
    """
    pass

