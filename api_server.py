"""MBTI API 服务启动入口。

Usage:
    python api_server.py                   # 默认 0.0.0.0:8000
    python api_server.py --port 3000       # 指定端口
    python api_server.py --reload          # 开发模式（热重载）
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "src.app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
