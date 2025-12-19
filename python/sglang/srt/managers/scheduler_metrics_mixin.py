from __future__ import annotations

import logging
import os
import time
from collections import defaultdict


from typing import TYPE_CHECKING, List, Optional
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
import json

from fastapi import FastAPI, Request
import uvicorn
from loguru import logger as logger_my

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Req, ScheduleBatch
from sglang.srt.metrics.collector import SchedulerMetricsCollector, SchedulerStats
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")


class KvMetrics:
    def __init__(self):
        self.request_active_slots = None
        self.request_total_slots = None
        self.kv_active_blocks = None
        self.kv_total_blocks = None
        self.num_requests_waiting = None
        self.gpu_cache_usage_perc = None
        self.gpu_prefix_cache_hit_rate = None
        self.data_parallel_rank = None


class SchedulerMetricsMixin:
    def init_metrics(self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]):
        # TODO
        '''
        # 在启动命令上增加一个enable选项
        # 在启动命令上增加一个port 端口
        # 在lb的启动命令上需要增加这个端口
        # 启动服务，同时self.tp_rank 不是0 的话，不是用启动服务
            提供一个路由供lb请求，接收lb发过来的lb的port 以及 lb给当前这个节点设置的特定的ID
            提供一个路由供lb请求，心跳探索（心跳返回 ID，这个ID从上面获取）
            有一个线程固定时间上报还是 走下面的触发函数？ 走触发吧，固定时间的是不太准确的
        '''

        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0

        self.stats = SchedulerStats()

        if self.enable_metrics:
            engine_type = "unified"
            labels = {
                "model_name": self.server_args.served_model_name,
                "engine_type": engine_type,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
            }
            if dp_rank is not None:
                labels["dp_rank"] = dp_rank
            self.metrics_collector = SchedulerMetricsCollector(labels=labels)
        self.enable_load_to_lb = bool(os.getenv("ENABLE_LOAD_TO_LB"))
        if self.tp_rank == 0 and self.enable_load_to_lb:
            self._init_load_server()
            if self.server_args.disaggregation_mode == "decode":
                self._init_decode_post_thread()

    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):
        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.attn_dp_rank
            )

    def update_spec_metrics(self: Scheduler, bs: int, num_accepted_tokens: int):
        self.spec_num_total_accepted_tokens += num_accepted_tokens + bs
        self.spec_num_total_forward_ct += bs
        self.num_generated_tokens += num_accepted_tokens
    def _init_load_server(self):
        self.post_thread_pool = ThreadPoolExecutor(max_workers=4)

        self.session_id = None
        self.lb_port = None
        self.lb_host_ip = None
        app = FastAPI()
        parent = self

        @app.post("/set_port_id")  # set lb的port 以及ID
        async def update_string(request: Request):
            data = await request.json()
            parent.session_id = data.get("session_id", "")    # TODO 这个session id 是空的
            parent.lb_port = data.get("lb_port", "")
            parent.lb_host_ip = data.get("lb_host_ip", "")
            if parent.session_id is None or parent.lb_port is None:
                return {"status": "error", "message": "session_id and load_port are required"}
            else:
                return {"status": "ok"}

        @app.get("/load_health_check")
        async def load_health_check(session_id: str = None):
            if parent.session_id is None or parent.lb_port is None:  # 如果POD中途挂了，然后又连上了，就会出现这种情况，会导致负载信息无法上报
                return {"status": "error", "message": "Not finish HandShake"}
            elif session_id is None:
                return {"status": "error", "message": "session_id is required"}
            else:
                return {"status": "ok"}

        def run():
            load_to_lb_port = os.getenv("LOAD_TO_LB_PORT")
            if load_to_lb_port is None:
                load_to_lb_port = 9999  # 默认9999 但是可能会引起冲突
            else:
                load_to_lb_port = int(load_to_lb_port)
            uvicorn.run(app, host="127.0.0.1", port=load_to_lb_port, log_level="debug")

        thread = threading.Thread(target=run, daemon=True)
        thread.start()


    def _init_decode_post_thread(self):
        self.decode_status_data = {
                "session_id": self.session_id,
                "type_pod": "Decode",
                "num_running_reqs": -1,
                "waiting_queue": 0,
                "pre_allocated_usage": "0",
                "retracted_req": 0,
                "cuda_graph": False,
                "gen_throughput": 0,
            }
        def run():
            while True:
                self.post_thread_pool.submit(self.post_info_to_lb, self.decode_status_data)
                time.sleep(1) # TODO 环境变量
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def post_info_to_lb(self, load_info: dict):
        if load_info["session_id"] is None:
            return
        if self.lb_host_ip is None or self.lb_port is None:
            logger_my.info("Not finish HandShake ")
        url = f"http://{self.lb_host_ip}:{str(self.lb_port)}/pod_load"
        logger_my.info(f"post_info_to_lb: {load_info}")
        logger_my.info(f"post_info_to_lb: {url}")
        try:
            requests.post(url, json.dumps(load_info), timeout=0.5)
        except Exception as e:
            logger_my.error(f"post_info_to_lb: {e}")
            pass

    def log_prefill_stats(
        self: Scheduler,
        adder: PrefillAdder,
        can_run_list: List[Req],
        running_bs: int,
        running_bs_offline_batch: int,
    ):
        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_prefill_tokens / gap_latency
        self.last_prefill_tokens = adder.log_input_tokens

        # TODO: generalize this for various memory pools
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_usage_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        elif self.is_hybrid_gdn:
            (
                full_num_used,
                _,
                full_token_usage,
                mamba_usage,
                _,
                _,
                _,
                _,
            ) = self._get_mamba_token_info()
            num_used = full_num_used
            token_usage = full_token_usage
            token_usage_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"mamba usage: {mamba_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_usage_msg = f"token usage: {token_usage:.2f}, "

        f = (
            f"Prefill batch [{self.forward_ct + 1}], "
            f"#new-seq: {len(can_run_list)}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {running_bs}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            f += f"#prealloc-req: {len(self.disagg_prefill_bootstrap_queue.queue)}, "
            f += f"#inflight-req: {len(self.disagg_prefill_inflight_queue)}, "
            f += f"input throughput (token/s): {self.last_input_throughput:.2f}, "
        else:
            f += f"#running-req: {running_bs}, "
            f += f"#queue-req: {len(self.waiting_queue)}, "
        if self.enable_load_to_lb:
            data = {
                "session_id": self.session_id,
                "type_pod": "Prefill",
                "unbootstrapped_req": len(self.disagg_prefill_bootstrap_queue.queue),
                "waiting_queue": len(self.waiting_queue),
                "transferring_req": len(self.disagg_prefill_inflight_queue),
                "input_throughput": round(self.last_input_throughput, 2),
                "cached_token": adder.log_input_tokens,
                "new_token": adder.log_input_tokens,
            }
            self.post_thread_pool.submit(self.post_info_to_lb, data)

        logger.info(f)

        if self.enable_metrics:
            # Basics
            total_tokens = adder.log_input_tokens + adder.log_hit_tokens
            cache_hit_rate = (
                adder.log_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            self.stats.num_running_reqs = running_bs
            self.stats.num_running_reqs_offline_batch = running_bs_offline_batch
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = token_usage
            if self.is_hybrid:
                self.stats.swa_token_usage = swa_token_usage
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            self.stats.cache_hit_rate = cache_hit_rate

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )

            # Others
            self.calculate_utilization()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def log_decode_stats(
        self: Scheduler, can_run_cuda_graph: bool, running_batch: ScheduleBatch = None
    ):
        batch = running_batch or self.running_batch

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency

        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)
        num_running_reqs_offline_batch = 0

        # TODO: generalize this for various memory pools
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_usage_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"#swa token: {swa_num_used}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        elif self.is_hybrid_gdn:
            (
                full_num_used,
                mamba_used,
                full_token_usage,
                mamba_usage,
                _,
                _,
                _,
                _,
            ) = self._get_mamba_token_info()
            num_used = full_num_used
            token_usage = full_token_usage
            token_usage_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"mamba num: {mamba_used}, "
                f"mamba usage: {mamba_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_usage_msg = f"#token: {num_used}, token usage: {token_usage:.2f}, "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        msg = f"Decode batch [{self.forward_ct}], #running-req: {num_running_reqs}, {token_usage_msg}"

        if self.spec_algorithm.is_none():
            spec_accept_length = 0
            spec_accept_rate = 0
        else:
            spec_accept_length = (
                self.spec_num_total_accepted_tokens / self.spec_num_total_forward_ct
            )
            # Calculate acceptance rate: accepted tokens / total draft tokens
            total_draft_tokens = self.spec_num_total_forward_ct * (
                (self.server_args.speculative_num_steps or 0) + 1
            )
            spec_accept_rate = (
                self.spec_num_total_accepted_tokens / total_draft_tokens
                if total_draft_tokens > 0
                else 0
            )
            self.cum_spec_accept_length += self.spec_num_total_accepted_tokens
            self.cum_spec_accept_count += self.spec_num_total_forward_ct
            self.spec_num_total_accepted_tokens = self.spec_num_total_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2f}, "
        cache_hit_rate = 0.0

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens:.2f}, "
            msg += f"#prealloc-req: {len(self.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.disagg_decode_transfer_queue.queue)}, "
            msg += f"#retracted-req: {len(self.disagg_decode_prealloc_queue.retracted_queue)}, "

        msg += (
            f"{'cuda graph' if self.device == 'cuda' else 'cpu graph'}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        # 更新decode_status
        if self.enable_load_to_lb:
            self.decode_status_data = {
                "session_id": self.session_id,
                "type_pod": "Decode",
                "num_running_reqs": num_running_reqs,
                "waiting_queue": len(self.waiting_queue),
                "pre_allocated_usage": f"{self.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens:.2f}",
                "retracted_req": len(self.disagg_decode_prealloc_queue.retracted_queue),
                "cuda_graph": can_run_cuda_graph,
                "gen_throughput": round(self.last_gen_throughput, 2),
            }
        # self.post_thread_pool.submit(self.post_info_to_lb, data)

        logger.info(msg)
        if self.enable_metrics:
            # Basics
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_running_reqs_offline_batch = num_running_reqs_offline_batch
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = token_usage
            if self.is_hybrid:
                self.stats.swa_token_usage = swa_token_usage
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            self.stats.cache_hit_rate = cache_hit_rate

            # Speculative decoding
            self.stats.spec_accept_rate = spec_accept_rate
            self.stats.spec_accept_length = spec_accept_length

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )

            # Others
            self.calculate_utilization()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def _emit_kv_metrics(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.stats.num_running_reqs
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(
            self.stats.token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.stats.num_queue_reqs
        kv_metrics.gpu_cache_usage_perc = self.stats.token_usage
        kv_metrics.gpu_prefix_cache_hit_rate = self.stats.cache_hit_rate
        kv_metrics.data_parallel_rank = self.dp_rank if self.dp_rank is not None else 0

        if not self.send_metrics_from_scheduler.closed:
            self.send_metrics_from_scheduler.send_pyobj(kv_metrics)

    def _publish_kv_events(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

    def calculate_utilization(self):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            if (
                self.stats.max_running_requests_under_SLO is not None
                and self.stats.max_running_requests_under_SLO > 0
            ):
                self.stats.utilization = max(
                    self.stats.num_running_reqs
                    / self.stats.max_running_requests_under_SLO,
                    self.stats.token_usage / 0.9,
                )
