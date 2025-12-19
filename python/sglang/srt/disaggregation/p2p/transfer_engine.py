"""
created by yansiyu01@baidu.com at 2025/07/31
"""
import sys
import logging
from typing import List, Any
import cuda_p2p_transfer

from sglang.srt.utils import get_free_port

logger = logging.getLogger(__name__)


class _ImmediateDoneHandle:
    """ _ImmediateDoneHandle """
    def is_done(self) -> bool:
        """ is_done """
        return True
    def wait(self) -> None:
        """ wait """
        return

class _BatchHandle:
    """ _BatchHandle """
    def __init__(self, handles):
        """ __init__ """
        self._handles = list(handles or [])

    def is_done(self) -> bool:
        """ is_done """
        for h in self._handles:
            fn = getattr(h, "is_done", None)
            if fn is None or not fn():
                return False
        return True

    def wait(self) -> None:
        """ wait """
        for h in self._handles:
            fn = getattr(h, "wait", None)
            if fn is not None:
                fn()
            else:
                while not getattr(h, "is_done", lambda: True)():
                    pass
                    

class P2PTransferEngine:
    """
    P2P传输引擎封装
    """
    def __init__(self, hostname: str, physical_gpu_id: int):
        """
        初始化P2P传输引擎
        """
        self.physical_gpu_id = physical_gpu_id
        self._rpc_port = get_free_port()
        self.hostname = hostname
        self.session_id = f"{self.hostname}:{self._rpc_port}"
        try:
            self.p2p_engine = cuda_p2p_transfer.CudaP2PTransfer(physical_gpu_id)
            logger.info("P2P engine initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize P2P engine")
            raise RuntimeError(f"P2P engine initialization failed: {e}") from e
    
    def register_buffer(self, ptr):
        """
        注册缓冲区
        Args:
            ptr: 缓冲区指针
        Returns:
            缓冲区句柄 (bytes)
        """
        try:
            handle = self.p2p_engine.register_buffer(ptr)
            return handle
        except Exception as e:
            logger.error(f"Failed to register buffer at ptr={ptr}: {e}")
            raise RuntimeError(f"Buffer registration failed: {e}")

    def transfer(self,
                 src_ptr: int,
                 src_dev: int,
                 dst_handle: bytes,
                 dst_dev: int,
                 dst_offset: int,
                 length: int):
        """
        执行P2P传输
        Args:
            src_ptr: 源指针
            src_dev: 源设备ID
            dst_handle: 目标句柄
            dst_dev: 目标设备ID
            dst_offset: 目标偏移量
            length: 传输长度
        Returns:
            0表示成功，非0表示错误
        """
        try:
            h = self.p2p_engine.transfer(src_ptr, src_dev, dst_handle, dst_dev, dst_offset, length)
            if isinstance(h, int):
                if h != 0:
                    raise RuntimeError(f"P2P transfer failed with code {h}")
                return _ImmediateDoneHandle()
            return h
        except Exception as e:
            logger.exception(f"Transfer failed: {e}")
            raise

    def transfer_many(self,
                      src_ptrs: List[int],
                      src_devs: List[int],
                      dst_handles: List[bytes],
                      dst_devs: List[int],
                      dst_offsets: List[int],
                      lengths: List[int]):
        """
        批量传输；优先调用底层 C++ 的 transfer_many；否则回退为逐条 transfer
        返回一个带 is_done() 的批次句柄
        """
        try:
            if hasattr(self.p2p_engine, "transfer_many"):
                h = self.p2p_engine.transfer_many(src_ptrs, src_devs, dst_handles, dst_devs, dst_offsets, lengths)
                return h
            # 回退：逐条提交
            handles = []
            for sp, sd, dh, dd, off, ln in zip(src_ptrs, src_devs, dst_handles, dst_devs, dst_offsets, lengths):
                handles.append(self.transfer(sp, sd, dh, dd, off, ln))
            return _BatchHandle(handles)
        except Exception as e:
            logger.exception(f"transfer_many failed: {e}")
            raise
        
    def register_d_handle(self, dst_handle: bytes) -> int:
        """
        注册目标句柄到引擎
        Args:
            dst_handle: 目标句柄
        Returns:
            0表示成功，非0表示错误
        """
        try:
            result = self.p2p_engine.register_d_handle(dst_handle)
            if result != 0:
                logger.error(f"Failed to register destination handle {dst_handle.hex()} with code {result}")
            return result
        except Exception as e:
            logger.exception(f"Destination handle registration failed: {e}")
            return 1

    def get_session_id(self):
        """
        获取会话ID
        Returns:
            会话ID字符串
        """
        return self.session_id