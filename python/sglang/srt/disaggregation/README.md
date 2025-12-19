```
# LB
python3 -m sglang.srt.disaggregation.mini_lb --prefill http://127.0.0.1:6667  \
--decode http://127.0.0.1:6668  --host 0.0.0.0 --port 9001   --prefill-bootstrap-ports 6666  \
--prefill-load-to-lb-ports 9999  --decode-load-to-lb-ports 9998 --enable-pd-load-record

--enable-pd-load-record     打开pd负载上报功能
--prefill-load-to-lb-ports  prefill的上报服务与lb的通信端口
--decode-load-to-lb-ports   decode的上报服务与lb的通信端口
```

```
#P
ENABLE_LOAD_TO_LB=True LOAD_TO_LB_PORT=9999 \
python3 -m sglang.launch_server --model-path /models/Qwen/Qwen3-4B  --disaggregation-mode prefill --nnodes 1 \
--node-rank 0 --tp-size 1 --decode-log-interval 1  --page-size 32 --trust-remote-code  --watchdog-timeout 1000000  \
--mem-fraction-static 0.89 --chunked-prefill-size 8192 --enable-p2p-check --attention-backend fa3 --reasoning-parser \
qwen3  --port 6667  --disaggregation-transfer-backend p2p --disaggregation-bootstrap-port 6666  --base-gpu-id 0

ENABLE_LOAD_TO_LB=True 打开负载上报功能
LOAD_TO_LB_PORT=9999   上报服务与lb的通信端口为9999
```

```
#D

ENABLE_LOAD_TO_LB=True LOAD_TO_LB_PORT=9998 \
NCCL_P2P_LEVEL=SYS python3 -m sglang.launch_server --model-path /models/Qwen/Qwen3-4B --disaggregation-mode decode \
--nnodes 1 --node-rank 0 --tp-size 1 --decode-log-interval 1 --page-size 32  --trust-remote-code  \
--disable-radix-cache  --watchdog-timeout 1000000  --mem-fraction-static 0.89 --chunked-prefill-size 8192 \
--enable-p2p-check --attention-backend fa3 --port 6668 --disaggregation-transfer-backend p2p --base-gpu-id 4 \
--cuda-graph-max-bs 1

ENABLE_LOAD_TO_LB=True 打开负载上报功能
LOAD_TO_LB_PORT=9999   上报服务与lb的通信端口为9998
```

