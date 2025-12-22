from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

@app.post("/v1/Nexus/get_best_instance")
async def get_best_instance(request: Request):
    """
    处理Nexus API的POST请求
    """
    # 获取并解析请求体
    raw_body = await request.body()
    request_json = {}
    
    if raw_body:
        try:
            request_json = json.loads(raw_body)
        except:
            request_json = {"error": "无效的JSON格式"}
    
    # 打印JSON请求体
    print(f"📥 请求体: {json.dumps(request_json, ensure_ascii=False)}")
    
    # 返回响应
    response_data = {"worker_ids": ["grpc://localhost:8002","grpc://localhost:8000"]}
    
    # 打印响应状态和内容
    print(f"📤 响应状态: 200")
    print(f"📤 响应内容: {json.dumps(response_data, ensure_ascii=False)}")
    
    return response_data

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 FastAPI 服务，端口: 5000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )