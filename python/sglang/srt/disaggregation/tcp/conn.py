"""
created by yansiyu01@baidu.com at 2025/09
"""
from __future__ import annotations

import asyncio
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

from aiohttp import web
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Union, Tuple

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.tcp.transfer_engine import TCPTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    format_tcp_address,
    get_free_port,
    get_int_env_var,
    get_ip,
    get_local_ip_auto,
    is_valid_ipv6_address,
)

logger = logging.getLogger(__name__)


class TCPTransferError(Exception):
    """
    TCPTransferError 异常类，表示 TCP 传输过程中发生的错误
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
        返回 TCPTransferError 对象的字符串表示形式
        """
        return f"TCPTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


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


# decode
@dataclasses.dataclass
class TransferInfo:
    """
    Decode 接收 KVCache 所需的目标传输信息
    """
    room: int
    endpoint: str
    # dst_port: int # 容易混淆，进行拆分
    ctrl_port: int # 用于 send_aux / sync_status_to_decode_endpoint (ZMQ 控制)
    data_port: int # 用于 tcp_engine.transfer KV 数据
    tcp_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    is_dummy: bool
    dst_buffer_ids: List[str] # TCP：目标缓冲区的ID列表

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        """
        从 ZMQ 消息中解析并创建一个新的实例
        """
        # # msg[0] = TRANSFER_HEADER
        if msg[6] == b"" and msg[7] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_buffer_ids = []
        else:
            dst_kv_indices = np.frombuffer(msg[6], dtype=np.int32)
            dst_aux_index = int(msg[7].decode("ascii"))
            is_dummy = False
            buffer_ids_str = msg[9].decode("ascii") if len(msg) > 9 else ""
            dst_buffer_ids = buffer_ids_str.split(",") if buffer_ids_str else []

        return cls(
            room=int(msg[1].decode("ascii")),
            endpoint=msg[2].decode("ascii"),
            ctrl_port=int(msg[3].decode("ascii")),   # rank_port
            data_port=int(msg[4].decode("ascii")),   # tcp_port
            # dst_port=int(msg[3].decode("ascii")),
            tcp_session_id=msg[5].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=int(msg[8].decode("ascii")),
            is_dummy=is_dummy,
            dst_buffer_ids=dst_buffer_ids,
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    """
    KVCache 注册信息
    """
    room: str
    endpoint: str
    ctrl_port: int
    tcp_session_id: str
    dst_buffer_ids: List[str]
    dst_aux_ptrs: list[int]

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        """
        从 ZMQ 消息中解析并创建一个新的实例
        """
        buffer_ids_str = msg[5].decode("ascii") if len(msg) > 5 else ""
        dst_buffer_ids = buffer_ids_str.split(",") if buffer_ids_str else []
        return cls(
            room=str(msg[1].decode("ascii")),
            endpoint=msg[2].decode("ascii"),
            ctrl_port=int(msg[3].decode("ascii")),
            tcp_session_id=msg[4].decode("ascii"),
            dst_buffer_ids=dst_buffer_ids,
            dst_aux_ptrs=[msg[6][i*64:(i+1)*64] for i in range(len(msg[6])//64)],
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


class TCPKVManager(BaseKVManager):
    """
    TCPKVManager 类 - 使用TCP传输KVCache
    """
    AUX_DATA_HEADER = b"AUX_DATA"
    STATUS_HEADER = b"STATUS"          # 状态同步
    REGISTER_HEADER = b"REGISTER"      # KV args 注册
    TRANSFER_HEADER = b"TRANSFER"      # 传输请求

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        """
        初始化 TCPKVManager 
        """
        self.kv_args = args
        self.disaggregation_mode = disaggregation_mode
        self.server_args = server_args
        self.is_mla_backend = is_mla_backend
        self.local_ip = get_local_ip_auto()

        self.tcp_port = get_free_port()
        self.tcp_engine = TCPTransferEngine(self.local_ip, self.tcp_port, self.kv_args.gpu_id)
        self.decode_physical_gpu_ids = {}
        
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        
        if not server_args.enable_dp_attention and server_args.dp_size != 1:
            raise ValueError(
                "If dp_attention is not enabled, dp size must be 1 in disaggregation mode."
            )
        
        self.request_status = {}
        self.rank_port = get_free_port()
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self._bind_server_socket()
        # logger.info(f"[TCPKVManager] TCP data_port={self.tcp_port}, ZMQ control_port={self.rank_port}")
        self.buffer_ids = []
        self.register_buffer_to_engine()
        
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {} 
            self.start_prefill_thread()
            self._register_to_bootstrap()
            self.session_failures = defaultdict(int) 
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
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

            self.bootstrap_time_out = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 30 
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set) 
            self.connection_lock = threading.Lock() 
            self.heartbeat_interval = max( 
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0 
            )
            self.max_failures = max( 
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_decode_thread()
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.prefill_tp_size_table: Dict[str, int] = {} 
            self.prefill_dp_size_table: Dict[str, int] = {} 
            
        else:
            raise ValueError(f"Unsupported DisaggregationMode: {self.disaggregation_mode}")
        
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()
        # KVCache 传输统计，以 room (每个请求) 为 key，保存 { "kv_bytes": int, "kv_time_ms": float, "kv_tokens": int }
        self.kv_transfer_stats: Dict[int, Dict[str, float | int]] = {}
        self.kv_transfer_stats_lock = threading.Lock()
        # 默认为 0（不开启），设置环境变量 SGLANG_KVCACHE_LOG=1，则会打印日志
        self.enable_kvcache_log = get_int_env_var("SGLANG_KVCACHE_LOG", 0) 

    def register_buffer_to_engine(self):
        """
        将KV缓冲区注册到TCP引擎中
        """
        logger.info("Start registering KV buffers to TCP engines.")
        self.buffer_ids = []
        for i, ptr in enumerate(self.kv_args.kv_data_ptrs):
            buffer_size = self.kv_args.kv_data_lens[i]
            try:
                buffer_id = self.tcp_engine.register_buffer(ptr, buffer_size, is_device=True, device_id=self.kv_args.gpu_id)
                # logger.info(f"Registered buffer {i}: id={buffer_id}, ptr={ptr}, size={buffer_size}")
                self.buffer_ids.append(buffer_id)
            except Exception as e:
                logger.error(f"Failed to register buffer {i}: {e}")
        
    def _connect(self, endpoint: str, is_ipv6: bool = False):
        """
        连接到指定的ZeroMQ端点
        """
        socket = zmq.Context().socket(zmq.PUSH)
        if is_ipv6:
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(endpoint)
        return socket
    
    def _bind_server_socket(self):
        """
        绑定服务器socket
        """
        self.server_socket.bind(format_tcp_address(self.local_ip, self.rank_port))

    def start_prefill_thread(self):
        """
        启动预填充线程
        """
        self.rank_port = get_free_port()
        self._bind_server_socket()

        def bootstrap_thread():
            """
            预填充线程函数
            """
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                msg_type = waiting_req_bytes[0]
                if msg_type == TCPKVManager.REGISTER_HEADER:
                    room = waiting_req_bytes[1].decode("ascii")
                    tcp_session_id = waiting_req_bytes[4].decode("ascii")
                    physical_gpu_id = int(waiting_req_bytes[7].decode("ascii"))
                    self.decode_physical_gpu_ids[tcp_session_id] = physical_gpu_id

                    buffer_ids_str = waiting_req_bytes[5].decode("ascii")
                    dst_buffer_ids = buffer_ids_str.split(",") if buffer_ids_str else []

                    aux_ptrs = list(struct.unpack(f"{len(waiting_req_bytes[6]) // 8}Q", waiting_req_bytes[6]))
                    self.decode_kv_args_table[tcp_session_id] = KVArgsRegisterInfo(
                        room=room,
                        endpoint=waiting_req_bytes[2].decode("ascii"),
                        ctrl_port=int(waiting_req_bytes[3].decode("ascii")),
                        tcp_session_id=tcp_session_id,
                        dst_buffer_ids=dst_buffer_ids,
                        dst_aux_ptrs=aux_ptrs,
                    )
                elif msg_type == TCPKVManager.TRANSFER_HEADER:
                    # 传输请求
                    required_dst_info_num = int(waiting_req_bytes[8].decode("ascii"))
                    room = int(waiting_req_bytes[1].decode("ascii"))
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}
                    
                    key = waiting_req_bytes[5].decode("ascii")
                    self.transfer_infos[room][key] = TransferInfo.from_zmq(waiting_req_bytes)
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)
        
        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def send_kvcache(
        self,
        req: TransferInfo,  
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        使用TCP发送 KVCache
        """
        try:
            if len(prefill_kv_indices) == 0 or len(dst_kv_indices) == 0:
                logger.error(f"No KV indices to transfer for session {req.tcp_session_id}")
                return 1

            prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
                prefill_kv_indices, dst_kv_indices
            )
            
            futures = []
            bytes_list = []      # 每个块的字节数
            start_time = time.perf_counter()  # 传输开始时间
            total_tokens = 0 

            for layer_id in range(len(self.kv_args.kv_data_ptrs)):
                for prefill_block, dst_block in zip(prefill_kv_blocks, dst_kv_blocks):
                    item_len = self.kv_args.kv_item_lens[layer_id]
                    src_offset = int(prefill_block[0]) * item_len
                    dst_offset = int(dst_block[0]) * item_len
                    length = len(prefill_block) * item_len

                    buffer_size = self.kv_args.kv_data_lens[layer_id]
                    if src_offset + length > buffer_size:
                        logger.error(f"Source offset + length exceeds buffer size for layer {layer_id}")
                
                    transfer_status = self._transfer_kv_block(
                        layer_id,
                        src_offset,
                        req.endpoint.split(":")[0],  # 远程地址
                        req.data_port,  # 远程端口
                        req.dst_buffer_ids[layer_id], # 目标缓冲区ID
                        dst_offset,
                        length,
                        )
                    futures.append(transfer_status)
                    bytes_list.append(length)
                    total_tokens += len(prefill_block) # 每个 block 的 token 数
            
            while futures:
                remaining = [task_transfer_statu for task_transfer_statu in futures \
                                if not task_transfer_statu.is_done()]
                futures = remaining
                if futures:
                    time.sleep(0.002)
            
            total_time_ms = (time.perf_counter() - start_time) * 1000.0  # 传输总耗时(ms)
            total_bytes = sum(bytes_list)  # 总字节数
            with self.kv_transfer_stats_lock: # 保存统计信息
                self.kv_transfer_stats[req.room] = {
                    "kv_bytes": total_bytes,
                    "kv_time_ms": total_time_ms,
                    "kv_tokens": total_tokens,
                }
    
            return 0  
        except Exception as e:
            logger.error(f"TCP KV cache transfer failed: {e}")
            return 1  

    def _transfer_kv_block(
        self,
        layer_id: int,
        src_offset: int,
        remote_addr: str,
        remote_port: int,
        dst_buffer_id: str,
        dst_offset: int,
        length: int
    ):
        """
        TCP 传输 KV 块
        """
        try:
            buffer_size = self.kv_args.kv_data_lens[layer_id]
            item_len = self.kv_args.kv_item_lens[layer_id]
            src_ptr = self.kv_args.kv_data_ptrs[layer_id] + src_offset # 源地址

            transfer_status_result = self.tcp_engine.transfer(
                src_ptr, 
                remote_addr,
                remote_port,
                dst_buffer_id,
                dst_offset,
                length,
                src_is_device=True,
                src_device_id=self.kv_args.gpu_id
            )
            return transfer_status_result
        except Exception as e:
            logger.exception(f"KV block transfer exception on layer {layer_id}: {e}")
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
        overall_ret = 0
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            ret = self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                ctrl_port=req.ctrl_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )
        if ret != 0:
            overall_ret = 1
        return overall_ret

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        ctrl_port: int,
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
                format_tcp_address(remote, ctrl_port), is_ipv6=is_valid_ipv6_address(remote)
            )

            socket.send_multipart(
                [
                    TCPKVManager.AUX_DATA_HEADER,
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

    def sync_status_to_decode_endpoint(
        self, remote: str, ctrl_port: int, room: int, status: int
    ):
        """
        同步状态
        """
        self._connect(
            format_tcp_address(remote, ctrl_port), is_ipv6=is_valid_ipv6_address(remote)
        ).send_multipart(
            [
                TCPKVManager.STATUS_HEADER,
                str(room).encode("ascii"),
                str(status).encode("ascii"),
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
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if req.tcp_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote tcp session {req.tcp_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.ctrl_port,
                                    req.room,
                                    KVPoll.Failed,
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
                        ret = self.send_kvcache(
                            req,
                            kv_chunk.prefill_kv_indices,
                            chunked_dst_kv_indice,
                            executor,
                        )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.tcp_session_id] += 1
                                if self.session_failures[req.tcp_session_id] >= 1:
                                    self.failed_sessions.add(req.tcp_session_id)
                                    logger.error(
                                        f"Session {req.tcp_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.data_port}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint, req.ctrl_port, req.room, KVPoll.Failed
                            )
                            break

                        if kv_chunk.is_last:
                            # Only the last chunk we need to send the aux data
                            ret = self.send_aux(
                                req,
                                kv_chunk.prefill_aux_index,
                                self.decode_kv_args_table[
                                    req.tcp_session_id
                                ].dst_aux_ptrs,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.ctrl_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, ctrl_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, ctrl_port, room, status
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
        self.rank_port = get_free_port()
        self._bind_server_socket()

        def decode_thread():
            """
            解码线程 
            """
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    msg_type = msg[0]
                    if msg_type == TCPKVManager.AUX_DATA_HEADER:
                        self._handle_aux_data(msg)
                        continue

                    elif msg_type == TCPKVManager.STATUS_HEADER:
                        bootstrap_room = int(msg[1].decode("ascii"))
                        status = int(msg[2].decode("ascii"))
                        if status == KVPoll.Failed:
                            self.record_failure(
                                bootstrap_room,
                                f"Failed to get kvcache from prefill instance, it might be dead",
                            )
                        self.update_status(bootstrap_room, status)
                    else:
                        logger.error(f"Unexpected message format: {msg}")

                except Exception as e:
                    logger.exception(f"Error in decode_thread: {e}")

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
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
            )
        )

    def check_status(self, bootstrap_room: int):
        """
        检查状态
        """
        return self.request_status.get(bootstrap_room, KVPoll.Bootstrapping)

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
        return self.tcp_engine.get_session_id() 

    def _register_to_bootstrap(self):
        """
        注册到 bootstarp 服务器
        """
        if self.dist_init_addr:
            ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0])
        else:
            ip_address = get_ip()

        bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Prefill",
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
            "engine_rank": self.kv_args.engine_rank,
        }

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Prefill instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Prefill instance failed to register to bootstrap server: {e}"
            )

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


class TCPKVSender(BaseKVSender):
    """
    TCPKVSender 类
    """
    def __init__(
        self,
        mgr: TCPKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        """
        初始化函数
        """
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state = None
        self.init_time = None
        self.curr_idx = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        初始化函数
        """
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
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
                    if elapsed >= self.kv_mgr.bootstrap_time_out:
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
                "TCP transfer failed due to an unknown reason"
            )
        raise TCPTransferError(self.bootstrap_room, failure_reason) 


class TCPKVReceiver(BaseKVReceiver):
    """
    TCPKVReceiver 类
    """
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: TCPKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        """
        初始化函数
        """
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.conclude_state = None
        self.data_parallel_rank = data_parallel_rank

        if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table:
            self.prefill_tp_size, self.prefill_dp_size = (
                self._get_prefill_parallel_info_from_server()
            )
            if self.prefill_tp_size is None or self.prefill_dp_size is None:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            else:
                logger.debug(
                    f"Fetch prefill parallel info from [{self.bootstrap_addr}]: DP size:{self.prefill_dp_size}, TP size:{self.prefill_tp_size}"
                )
                self.kv_mgr.prefill_tp_size_table[self.bootstrap_addr] = (
                    self.prefill_tp_size
                )
                self.kv_mgr.prefill_dp_size_table[self.bootstrap_addr] = (
                    self.prefill_dp_size
                )
        else:
            self.prefill_tp_size = self.kv_mgr.prefill_tp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_dp_size = self.kv_mgr.prefill_dp_size_table[
                self.bootstrap_addr
            ]

        local_tp_size_per_dp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        prefill_tp_size_per_dp_rank = self.prefill_tp_size // self.prefill_dp_size
        if local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            )
            self.required_dst_info_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        elif local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            ) // (local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank)
            self.required_dst_info_num = (
                local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
            )
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"

            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank + 1)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                )
            ]

            self.target_tp_rank = self.target_tp_ranks[0]
            self.required_dst_info_num = 1

        if self.data_parallel_rank is not None:
            logger.debug(f"Targeting DP rank: {self.data_parallel_rank}")
            self.target_dp_group = self.data_parallel_rank
        else:
            self.target_dp_group = bootstrap_room % self.prefill_dp_size

        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                bootstrap_info = self._get_bootstrap_info_from_server(
                    target_tp_rank,
                    self.target_dp_group,
                )
                if bootstrap_info is not None:
                    bootstrap_info["is_dummy"] = not bool(
                        target_tp_rank == self.target_tp_rank
                        or self.target_tp_rank is None
                    )
                    logger.debug(
                        f"Fetched bootstrap info: {bootstrap_info} for DP {self.target_dp_group} TP {target_tp_rank}"
                    )
                    bootstrap_infos.append(bootstrap_info)
                else:
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {self.target_dp_group}",
                    )
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    return

            self.bootstrap_infos = bootstrap_infos
            self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

            self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(self, engine_rank, target_dp_group):
        """
        从服务器获取引导信息
        """
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
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

    def _get_prefill_parallel_info_from_server(self) -> Tuple[int, int]:
        """
        从服务器获取预填充并行信息
        """
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return int(prefill_parallel_info["prefill_tp_size"]), int(
                    prefill_parallel_info["prefill_dp_size"]
                )
            else:
                logger.error(
                    f"Failed to get prefill parallel info: {response.status_code}, {response.text}"
                )
                return None, None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None, None

    def _register_kv_args(self):
        """
        注册 KV 参数
        """
        tp_rank = self.kv_mgr.kv_args.engine_rank
        physical_gpu_id = (
            self.kv_mgr.server_args.base_gpu_id +
            tp_rank * self.kv_mgr.server_args.gpu_id_step
        )
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.info(f"Connecting to prefill server at {self.prefill_server_url}")
            
            buffer_ids_str = ",".join(self.kv_mgr.buffer_ids) if self.kv_mgr.buffer_ids else ""
            
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            gpu_id_str = str(physical_gpu_id)
            
            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        TCPKVManager.REGISTER_HEADER,
                        "None".encode("ascii"), 
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        buffer_ids_str.encode("ascii"),
                        packed_aux_data_ptrs,
                        gpu_id_str.encode("ascii"),
                    ]
                )
    
    @classmethod
    def _connect(cls, endpoint: str):
        """
        连接到指定的ZMQ端点
        """
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        """
        初始化函数，用于连接到预填充服务器并发送初始化请求
        """
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            is_dummy = bootstrap_info["is_dummy"]

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                buffer_ids_str = ",".join(self.kv_mgr.buffer_ids) if not is_dummy else ""
                sock.send_multipart(
                    [
                        TCPKVManager.TRANSFER_HEADER,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"), # ctrl_port
                        str(self.kv_mgr.tcp_port).encode("ascii"),  # data_port 
                        # str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        str(self.required_dst_info_num).encode("ascii"),
                        buffer_ids_str.encode("ascii") if not is_dummy else b"", 
                    ]
                )

    def poll(self) -> KVPoll:
        """
        轮询检查状态
        """
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        """
        清除
        """
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        """
        失败异常
        """
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "TCP transfer failed due to an unknown reason"
            )
        raise TCPTransferError(self.bootstrap_room, failure_reason)


class TCPKVBootstrapServer(BaseKVBootstrapServer):
    """
    TCPKVBootstrapServer 类
    """
    def __init__(self, port: int):
        """
        初始化
        """
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.tp_size = None
        self.dp_size = None
        self.tp_size_per_dp_rank = None
        self.prefill_port_table: Dict[int, Dict[int, Dict[str, Union[str, int]]]] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        """
        运行线程 
        """
        self.thread.start()

    def _setup_routes(self):
        """
        设置路由
        """
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        """
        处理健康检查请求
        """
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        """
        处理HTTP请求
        """
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        """
        处理 PUT 请求的异步函数
        """
        data = await request.json()
        role = data["role"]
        tp_size = data["tp_size"]
        dp_size = data["dp_size"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        engine_rank = int(data["engine_rank"])

        if self.tp_size is None:
            self.tp_size = tp_size

        if self.dp_size is None:
            self.dp_size = dp_size

        tp_size_per_dp_rank = tp_size // dp_size
        if self.tp_size_per_dp_rank is None:
            self.tp_size_per_dp_rank = tp_size_per_dp_rank

        if role == "Prefill":
            dp_group = engine_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank

            async with self.lock:
                if dp_group not in self.prefill_port_table:
                    self.prefill_port_table[dp_group] = {}

            self.prefill_port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register prefill bootstrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        """
        处理 GET 请求的异步函数
        """
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        if not engine_rank or not target_dp_group:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        if int(engine_rank) == -1 and int(target_dp_group) == -1:
            prefill_parallel_info = {
                "prefill_tp_size": self.tp_size,
                "prefill_dp_size": self.dp_size,
            }
            return web.json_response(prefill_parallel_info, status=200)

        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(target_dp_group)][
                int(engine_rank)
            ]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    def _run_server(self):
        """
        启动服务
        """
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """
        关闭服务
        """
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
