"""
Enterprise RAG Dialogue System - Main Entry Point
==================================================
主程序入口，提供命令行交互界面。

使用方式：
    # 启动交互式对话
    python main.py

    # 单次查询
    python main.py --query "什么是 RAG？"

    # 指定会话 ID
    python main.py --session-id my_session
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from loguru import logger


def setup_logging(log_level: str = "INFO") -> None:
    """配置日志输出格式"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{extra[module]}</cyan> - {message}",
        level=log_level,
        colorize=True,
    )
    logger.add(
        "logs/rag_system.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[module]} - {message}",
    )


def check_environment() -> bool:
    """检查必要的环境变量"""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        logger.warning(
            "未检测到 OPENAI_API_KEY 环境变量。\n"
            "系统将使用 Mock 检索，LLM 调用可能失败。\n"
            "请在 .env 文件中设置 LLM_API_KEY=your_api_key"
        )
        return False
    return True


def interactive_chat(workflow, session_id: str) -> None:
    """
    交互式对话循环。
    支持多轮对话，维护会话记忆。
    """
    print("\n" + "="*60)
    print("  企业级 RAG 对话系统  v1.0.0")
    print("  基于 LangGraph + LlamaIndex + GraphRAG + DSPy + RAGAS")
    print("="*60)
    print(f"  会话 ID: {session_id}")
    print("  输入 'exit' 退出，'clear' 清除记忆，'history' 查看历史")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("用户: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        elif user_input.lower() == "exit":
            print("再见！")
            break
        elif user_input.lower() == "clear":
            workflow.clear_session(session_id)
            print("系统: 记忆已清除\n")
            continue
        elif user_input.lower() == "history":
            history = workflow.get_session_history(session_id)
            if not history:
                print("系统: 暂无历史记录\n")
            else:
                print("历史记录:")
                for role, content in history[-10:]:
                    prefix = "用户" if role == "human" else "助手"
                    print(f"  {prefix}: {content[:100]}")
                print()
            continue

        # 执行 RAG 查询
        print("系统: 正在处理...", end="\r")
        result = workflow.run(user_input, session_id=session_id)

        answer = result.get("answer", "（无法生成答案）")
        eval_scores = result.get("evaluation_scores", {})
        error = result.get("error")

        if error:
            print(f"系统: [错误] {error}\n")
        else:
            print(f"助手: {answer}")
            if eval_scores:
                scores_str = " | ".join(f"{k}: {v:.2f}" for k, v in eval_scores.items())
                print(f"      [评估: {scores_str}]")
            print()


def single_query(workflow, query: str, session_id: str) -> None:
    """执行单次查询并输出结果"""
    print(f"查询: {query}")
    print("处理中...")
    result = workflow.run(query, session_id=session_id)

    print(f"\n答案:\n{result.get('answer', '（无法生成答案）')}")

    if result.get("reasoning_steps"):
        print(f"\n推理步骤:")
        for i, step in enumerate(result["reasoning_steps"], 1):
            print(f"  {i}. {step[:200]}")

    if result.get("evaluation_scores"):
        print(f"\n评估分数:")
        for metric, score in result["evaluation_scores"].items():
            print(f"  {metric}: {score:.3f}")

    if result.get("error"):
        print(f"\n错误: {result['error']}")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="企业级 RAG 对话系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", type=str, help="单次查询模式")
    parser.add_argument("--session-id", "-s", type=str, help="指定会话 ID")
    parser.add_argument("--log-level", "-l", default="WARNING", help="日志级别")
    parser.add_argument(
        "--index-dir", "-i", type=str, help="文档目录（启动时加载到向量索引）"
    )
    args = parser.parse_args()

    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    setup_logging(args.log_level)

    # 检查环境
    check_environment()

    # 初始化工作流
    from rag_system.workflow.graph import RAGWorkflow

    print("正在初始化 RAG 系统...")
    workflow = RAGWorkflow()

    # 可选：加载文档到向量索引
    if args.index_dir:
        from rag_system.data.index_manager import IndexManager
        print(f"正在加载文档: {args.index_dir}")
        manager = IndexManager()
        count = manager.load_documents_from_directory(args.index_dir)
        print(f"已加载 {count} 个文档")

    session_id = args.session_id or str(uuid.uuid4())

    # 执行查询
    if args.query:
        single_query(workflow, args.query, session_id)
    else:
        interactive_chat(workflow, session_id)


if __name__ == "__main__":
    main()
