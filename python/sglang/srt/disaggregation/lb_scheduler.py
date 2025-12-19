#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
-------------------------------------------------
   File Name:     sglang/python/sglang/srt/disaggregation/lb_scheduler.py
   Description:   不同的调度策略
   Author:        wuzhensheng01@baidu.com
   date:          2025/09/15
   project:       None
-------------------------------------------------
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from enum import Enum
import time
import requests

from loguru import logger as logger_my

from sglang.srt.disaggregation.safedict import AutoSafeDict, PodLoadInfo


class ReturnCode(Enum):
    SUCCESS = 0  # 正确
    ERROR_UNKNOWN = 1  # 未知错误
    ERROR_TIMEOUT = 2  # 超时
    ERROR_NETWORK = 3  # 网络错误
    ERROR_INVALID = 4  # 参数无效
    ERROR_NOTEXIST = 5  # 不存在


class DecodeNode:
    """
    Decode 的具体节点
    """

    def __init__(self, ip_address: str, load_port: int, server_port: int):
        self.ip_address = ip_address
        self.server_port = server_port
        self.load_port = load_port
        self.status_alive = True  # 该节点是否正常工作
        self.disconnected_count = 0
        self.alive_url = f'http://{self.ip_address}:{str(self.load_port)}/load_health_check'
        self.pod_status = PodStaus()
        self.rw_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.heart_thread = threading.Thread(target=self.heartbeat_test, daemon=True)
        self.heart_thread.start()

    def stop(self):  #
        self._stop_event.set()
        self.heart_thread.join()

    def heartbeat_test(self):
        while not self._stop_event.is_set():
            session_id = f"{self.ip_address}:{str(self.load_port)}"
            params = {"session_id": session_id}
            try:
                response = requests.get(self.alive_url, params=params, timeout=1)
                data = response.json()
                status_value = data.get("status")
                if status_value == "error":
                    message = data.get("message")
                    if message == "Not finish HandShake":  # 没有完成握手，需要重连的
                        logger_my.info(
                            "Node:{}:{} load_port:{} Not finish HandShake  try to reconnect!".format(self.ip_address,
                                                                                                     self.server_port,
                                                                                                     self.load_port))
                        reconnect_result = self.set_lb_port_reconnect()  # 重连
                        if reconnect_result == 0:
                            self.status_alive = True
                            self.disconnected_count = 0
                        else:
                            self.disconnected_count += 1
                            self.status_alive = False
                else:
                    self.status_alive = True
            except Exception as e:
                self.status_alive = False
                self.disconnected_count += 1
                if hasattr(self, "bootstrap_port"):
                    logger_my.info("Prefill Node: {} Disconnected, load_port:{}".format(
                        self.ip_address + ":" + str(self.server_port), self.load_port))
                else:
                    logger_my.info("Decode Node: {} Disconnected, load_port:{}".format(
                        self.ip_address + ":" + str(self.server_port), self.load_port))
            time.sleep(1)

    def set_lb_port_reconnect(self):
        session_id = f"{self.ip_address}:{str(self.load_port)}"  # 发送给POD，后面POD上报信息的时候，是要用这个参数的
        url = f"http://{self.ip_address}:{str(self.load_port)}/set_port_id"
        payload = {
            "session_id": session_id,
            "lb_port": int(lb_port),
            "lb_host_ip": str(lb_host_ip)
        }
        try:
            response = requests.post(url, json=payload, timeout=1)
            data = response.json()
            if data.get("status") != "ok":
                logger_my.info("set lb port to node Error:{}".format(data.get("message")))
                return ReturnCode.ERROR_NETWORK
            else:
                logger_my.info("Reconnet Node:{}:{} load_port:{} Success".format(self.ip_address,
                                                                                 self.server_port,
                                                                                 self.load_port))
                return ReturnCode.SUCCESS

        except requests.exceptions.RequestException as e:
            logger_my.info("请求失败:", e)
            return ReturnCode.ERROR_UNKNOWN

    def set_info(self, pod_load_info):
        with self.rw_lock:
            #logger_my.info("pod_load_info: {}".format(pod_load_info))
            self.pod_status.num_running_reqs = pod_load_info["num_running_reqs"]
            self.pod_status.pre_allocated_usage = pod_load_info["pre_allocated_usage"]
            self.pod_status.retracted_req = pod_load_info["retracted_req"]
            self.pod_status.waiting_queue = pod_load_info["waiting_queue"]
            self.pod_status.cuda_graph = pod_load_info["cuda_graph"]
            self.pod_status.gen_throughput = pod_load_info["gen_throughput"]
            self.pod_status.cla_scores("decode")


class PrefillNode(DecodeNode):
    def __init__(self,
                 ip_address: str,
                 load_port: int,
                 server_port: int,
                 bootstrap_port: int):
        super().__init__(ip_address,
                         load_port,
                         server_port)
        self.bootstrap_port = bootstrap_port

    def set_info(self, pod_load_info):
        self.rw_lock.acquire()
        self.pod_status.unbootstrapped_req = pod_load_info["unbootstrapped_req"]
        self.pod_status.waiting_queue = pod_load_info["waiting_queue"]
        self.pod_status.input_throughput = pod_load_info["input_throughput"]
        self.pod_status.cached_token = pod_load_info["cached_token"]
        self.pod_status.new_token = pod_load_info["new_token"]
        self.pod_status.transferring_req = pod_load_info["transferring_req"]
        self.rw_lock.release()
        self.pod_status.cla_scores("prefill")


class DeployNode:  # 部署PD的 Node node里面有P和D节点，至少有一个P和D
    def __init__(self, ip_address: str, timeout_time: int = 30):
        self.ip_address = ip_address  # 每个node的IP 是确定的
        self.p_pod = AutoSafeDict()
        self.d_pod = AutoSafeDict()
        self.timeout = timeout_time
        threading.Thread(target=self.delete_node, daemon=True).start()

    def find_node(self, session_id):
        with self.p_pod.read():
            return session_id in self.p_pod or session_id in self.d_pod

    def delete_node(self):
        while True:
            to_delete = list()
            for session_id, pod in self.p_pod.items():
                if pod.disconnected_count > self.timeout:
                    to_delete.append(session_id)
            for session_id in to_delete:
                try:
                    self.p_pod[session_id].stop()
                    with self.p_pod.write():
                        del self.p_pod[session_id]
                except Exception as e:
                    logger_my.info("Error:{}".format(e))
            to_delete.clear()

            for session_id, pod in self.d_pod.items():
                if pod.disconnected_count > self.timeout:
                    to_delete.append(session_id)
                else:
                    pass
            for session_id in to_delete:
                self.d_pod[session_id].stop()
                with self.d_pod.write():
                    del self.d_pod[session_id]
            to_delete.clear()
            time.sleep(0.1)

    def add_node(self,
                 ip_address: str,
                 load_port: int,
                 server_port: int,
                 bootstrap_port: int = None):
        if ip_address != self.ip_address:
            return ReturnCode.ERROR_INVALID  # 无效IP
        session_id = f"{ip_address}:{load_port}"
        if self.find_node(session_id):
            return 0
        if bootstrap_port is None:
            with self.d_pod.read():
                if session_id not in self.d_pod:
                    with  self.d_pod.write():
                        self.d_pod[session_id] = DecodeNode(ip_address,
                                                            load_port,
                                                            server_port)
                else:
                    return ReturnCode.ERROR_NOTEXIST
        else:
            with self.p_pod.read():
                if session_id not in self.p_pod:
                    with self.p_pod.write():
                        self.p_pod[session_id] = PrefillNode(ip_address,
                                                             load_port,
                                                             server_port,
                                                             bootstrap_port)
                else:
                    return ReturnCode.ERROR_NOTEXIST
        return ReturnCode.SUCCESS

    def add_node_info(self, pod_load_info):
        type_pod = pod_load_info["type_pod"]  # "Prefill" or "Decode"
        if type_pod == "Prefill":
            session_id = pod_load_info["session_id"]
            if session_id not in self.p_pod:
                logger_my.info("session_id:{} not in Node ".format(session_id))
            else:
                self.p_pod[session_id].set_info(pod_load_info)
        elif type_pod == "Decode":
            session_id = pod_load_info["session_id"]
            if session_id not in self.d_pod:
                logger_my.info("session_id:{} not in Node ".format(session_id))
            else:
                self.d_pod[session_id].set_info(pod_load_info)
        else:
            logger_my.info("type error")


class PodStaus:
    def __init__(self):
        self.unbootstrapped_req: int = 0  # 处理握手队列队列的长度
        self.waiting_queue: int = 0  # 等待队列的长度
        self.transferring_req: int = 0  # 处于KvCache传输的队列长度
        self.input_throughput: int = 0  # prefill当前的吞吐
        self.cached_token: int = 0  # 有缓存的token
        self.new_token: int = 0  # 新产生的token

        self.num_running_reqs: int = 0  # Decode阶段当前的推理Batch
        self.pre_allocated_usage: float = 0.0  # disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens
        self.retracted_req: int = 0
        self.cuda_graph: bool = False
        self.gen_throughput = 0

        self.scores = 0
        self.scores_lock = threading.Lock()

    def cla_scores(self, type: str) -> float:
        """
        握手队列的长度 一个+2分， 等待队列的长度一个 + 3分，传输的长度一个 +1 分 按照总分从低到高进行排序，取最小优先
        """
        self.scores_lock.acquire()
        if type == "prefill":
            self.scores = self.transferring_req + 2 * self.unbootstrapped_req + 3 * self.waiting_queue
        elif type == "decode":
            self.scores = self.num_running_reqs + int(self.cuda_graph == False) * 10
        else:
            raise NotImplementedError
        self.scores_lock.release()


class LbScheduler:
    def __init__(self, max_workers=8):
        self.max_workers = max_workers
        # 在线程池只初始化一次，后续反复复用
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def select_p_d_pod(self, node_dict: AutoSafeDict):
        """
        遍历 node_dict，把满足条件的 pod 提交到线程池，
        等所有任务完成后，收集结果并返回。
        """
        result = []
        future_to_session = {}

        # 提交任务
        for session_id in node_dict.keys():
            future = self.executor.submit(self.select_from_node, node_dict[session_id])
            future_to_session[future] = session_id

        # 等待所有任务完成，收集结果
        for future in future_to_session:
            session_id = future_to_session[future]
            try:
                res = future.result()  # 这里会阻塞，直到该任务完成
                result.append(res)
            except Exception as e:
                logger_my.info(f"[ERROR] select_from_node(session_id={session_id}) failed: {e}")
        # result [[p_pod, d_pod], [p_pod, d_pod], [p_pod, d_pod] .... None .... [p_pod, d_pod]]
        result = [x for x in result if x is not None]
        if len(result) == 0:
            return None
        else:
            result.sort(key=lambda x: x[0].pod_status.scores)
            return result[0]

    def select_from_node(self, node: DeployNode):
        # 从一个node里面选出一个P和一个D，目前因为是机内P2P的PD，所以P和D只能在同一个机器内
        """
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

        return prefill_config.url, prefill_config.bootstrap_port, decode_server
        返回值和原来保持一致
        """
        if len(node.p_pod) == 0 or len(node.d_pod) == 0:
            # 当前P2P的PD分离只支持P和D在同一个节点上
            return None

        pd_pod_result = list()
        future_p = self.executor.submit(self.select_pod, node.p_pod, "prefill") #P
        future_d = self.executor.submit(self.select_pod, node.d_pod, "decode")
        pd_pod_result.append(future_p.result())
        pd_pod_result.append(future_d.result())
        return pd_pod_result # [p_pod, d_pod] # TODO 可能存在一个是None



    def select_pod(self, pod_dict: DeployNode, select_type: str):
        """
        握手队列的长度 一个+2分， 等待队列的长度一个 + 3分，传输的长度一个 +1 分 按照总分从低到高进行排序，取最小优先
        """
        # current_lowest_scores = 99999
        # current_server_ip = None
        # current_bootstrap_port = None

        current_lowest_scores = 99999
        pod = None
        for key in pod_dict.keys():
            if pod_dict[key].pod_status.scores < current_lowest_scores and pod_dict[key].status_alive: # pod必须是alive的
                current_lowest_scores = pod_dict[key].pod_status.scores
                pod = pod_dict[key]
        return pod

