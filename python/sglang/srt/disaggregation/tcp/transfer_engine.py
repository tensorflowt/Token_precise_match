"""
created by yansiyu01@baidu.com at 2025/09
"""
import sys
import logging
import tcp_transfer_engine
from typing import Optional

from sglang.srt.utils import get_free_port

logger = logging.getLogger(__name__)


class TCPTransferEngine:
    """
    TCP传输引擎封装，支持GPU间数据传输
    """
    def __init__(self, local_ip: str, port: Optional[int] = None, dev_id: int = 0):
        """
        初始化TCP传输引擎
        Args:
            local_ip: IP地址
            port: 端口号
            dev_id: 本rank使用的GPUID
        """
        self.local_ip = local_ip
        self.port = port if port is not None else get_free_port()
        self.session_id = f"{self.local_ip}:{self.port}"
        self.dev_id = dev_id

        try:
            self.tcp_engine = tcp_transfer_engine.TCPTransferEngine(self.local_ip, self.port, self.dev_id)
            logger.info(f"TCP engine initialized successfully on {self.local_ip}:{self.port}, dev_id={self.dev_id}")
        except Exception as e:
            logger.exception("Failed to initialize TCP engine")
            raise RuntimeError(f"TCP engine initialization failed: {e}") from e
    
    def register_buffer(self, ptr: int, size: int, is_device: bool = True, device_id: int = -1) -> str:
        """
        注册缓冲区
        Args:
            ptr: host/device 内存地址（整数）
            size: 字节缓冲区大小
            is_device: True 表示传输 GPU 数据
            device_id: 如果是 device pointer, 指定 GPU id
        Returns:
            buffer_id：缓冲区ID (string)
        """
        try:
            buffer_id = self.tcp_engine.register_buffer(ptr, size, is_device, device_id)
            return buffer_id
        except Exception as e:
            logger.error(f"Failed to register buffer at ptr={ptr}, size={size}: {e}")
            raise RuntimeError(f"Buffer registration failed: {e}")

    def transfer(
        self,
        src_ptr: int,
        remote_addr: str,
        remote_port: int,
        dst_buffer_id: str,
        dst_offset: int,
        length: int,
        src_is_device: bool = True,
        src_device_id: int = -1,
    ):
        """
        执行TCP传输
        Args:
            src_ptr: 源GPU内存指针
            remote_addr: 远程主机地址
            remote_port: 远程端口
            dst_buffer_id: 目标 buffer id
            dst_offset: 目标偏移量
            length: 传输长度
            src_is_device: 源是否是 device pointer
            src_device_id: 源GPU设备ID
        Returns:
            TransferFuture对象
        """
        try:
            transfer_future = self.tcp_engine.transfer(
                src_ptr, 
                remote_addr,
                remote_port,
                dst_buffer_id,
                dst_offset,
                length,
                src_is_device,
                src_device_id
            )
            return transfer_future
        except Exception as e:
            logger.exception(f"Transfer failed: {e}")
            raise
    
    def get_session_id(self):
        """
        获取会话ID
        Returns:
            会话ID字符串
        """
        return self.session_id
        