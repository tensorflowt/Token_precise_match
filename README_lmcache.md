# 创建容器
```
docker run --name sgl_0.4.8_lmcache \
-it \
--gpus all \
--net=host \
--shm-size=48G \
--privileged=true \
-v /mnt/data1/zwt:/work \
--entrypoint /bin/bash \
vllm/vllm-openai:v0.8.0
```

# 源码编译
```
cd ～/sglang
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e "python[all]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 安装依赖lmcache
```
pip install lmcache==0.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 服务启动
```
CUDA_VISIBLE_DEVICES=0,1 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=200 \
LMCACHE_MAX_LOCAL_DISK_SIZE=400 \
LMCACHE_CONFIG_FILE="cpu-offload.yaml" \
LMCACHE_USE_EXPERIMENTAL=True \
nohup \
python3 -m sglang.launch_server \
    --model-path /work/models/Qwen/Qwen3-30B-A3B-FP8 \
    --enable-torch-compile \
    --served-model-name qwen3-30b \
    --tp 2 \
    --port 30009 \
    --host 0.0.0.0 \
    --log-level info \
    --enable-metrics \
    --enable-p2p-check \
    --attention-backend fa3 \
    --enable-lmcache-connector > /work/exp/sgl_lmcache_exp/0729/server_sgl_lmcache.log 2>&1 &
```

#### cpu-offload.yaml
```
chunk_size: 256
local_cpu: true
max_local_cpu_size: 200
max_local_disk_size: 400.0
```


# 接口测试
```
curl -X POST "http://127.0.0.1:30009/v1/chat/completions" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer token-aicp" \
-d '{
      "model": "qwen3-30b",
      "messages": [
        {
          "role": "user",
          "content": "请详细介绍一下北京这座城市，包括历史、文化和现代发展。"
        }
      ],
      "max_tokens": 200
    }'
```

# 压测
```
python3 -m sglang.bench_serving \
--model /work/models/Qwen/Qwen3-30B-A3B-FP8 \
--backend sglang \
--num-prompts 512 \
--random-input-len 5120 \
--random-output-len 128 \
--dataset-name random \
--dataset-path /work/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 2 \
--seed 42 \
--host 127.0.0.1 \
--random-range-ratio 1.0 \
--port 30009
```