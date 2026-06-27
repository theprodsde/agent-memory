"""Basic SDK usage example."""

from agent_memory import Memory, MemoryAction, MemoryType


def main() -> None:
    memory = Memory(persist_dir=".agent_memory_demo")

    memory.remember(
        query="How do I reset my password?",
        response="Go to Settings → Security → Reset Password and follow the email link.",
        type=MemoryType.CONVERSATION,
        tags=["auth", "faq"],
    )

    memory.remember(
        query="Current API rate limit",
        response="1000 requests/minute per API key.",
        type=MemoryType.FACT,
        requires_verification=True,
    )

    queries = [
        "How do I reset my password?",
        "Give me a one-sentence explanation of password reset",
        "What is the API rate limit?",
        "What's the weather today?",
    ]

    for query in queries:
        decision = memory.resolve(query)
        print(f"\nQuery: {query}")
        print(f"Action: {decision.action.value} (confidence: {decision.confidence:.2f})")
        print(f"Reason: {decision.reason}")

        if decision.action == MemoryAction.REPLAY:
            print(f"→ Replay: {decision.response}")
        elif decision.action == MemoryAction.RESTORE:
            print("→ Restore context:")
            print(memory.format_restore_context(decision))
        elif decision.action == MemoryAction.VERIFY:
            print("→ Verify before reuse:")
            print(memory.format_verify_context(decision))
        else:
            print("→ Answer from scratch")


if __name__ == "__main__":
    main()
