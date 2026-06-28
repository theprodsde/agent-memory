from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_memory import Memory
from agent_memory.benchmark import format_benchmark_report, run_benchmark
from agent_memory.eval import EvalDataset, format_eval_report, run_eval_suite, seed_dataset


def _default_datasets_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmarks" / "datasets"


def _create_memory(args: argparse.Namespace) -> Memory:
    return Memory(persist_dir=args.data_dir, backend=args.backend)


def cmd_remember(args: argparse.Namespace) -> int:
    memory = _create_memory(args)
    entry = memory.remember(
        args.query,
        args.response,
        type=args.type,
        scope=args.scope,
        ttl=args.ttl,
    )
    print(f"Stored memory {entry.id}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    memory = _create_memory(args)
    decision = memory.resolve(args.query)
    print(decision)
    if args.explain:
        print()
        print(decision.explain())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    memory = _create_memory(args)
    stats = memory.stats()
    print("Agent Memory Stats")
    print("==================")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    memory = _create_memory(args)
    result = memory.cleanup(delete=args.delete)
    print(result)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    memory = _create_memory(args)

    if args.seed:
        for dataset_path in _default_datasets_dir().glob("*.json"):
            dataset = EvalDataset.load(dataset_path)
            seed_dataset(memory, dataset)
            for case in dataset.cases:
                args.queries.extend([case.query] * max(1, args.repeat))

    queries = args.queries or [
        "How do I reset my password?",
        "I forgot my password",
        "What is the API rate limit?",
        "What's the weather today?",
    ]

    if args.repeat > 1 and not args.seed:
        queries = [q for q in queries for _ in range(args.repeat)]

    result, comparison = run_benchmark(memory, queries, baseline_no_memory_ms=args.baseline_ms)
    print(format_benchmark_report(result, comparison))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    memory = Memory(persist_dir=args.data_dir)
    datasets_dir = Path(args.datasets) if args.datasets else _default_datasets_dir()

    if not datasets_dir.exists():
        print(f"Datasets directory not found: {datasets_dir}", file=sys.stderr)
        return 1

    paths = sorted(datasets_dir.glob("*.json"))
    if not paths:
        print(f"No datasets found in {datasets_dir}", file=sys.stderr)
        return 1

    results = run_eval_suite(memory, paths)
    print(format_eval_report(results))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-memory", description="Agent Memory CLI")
    parser.add_argument(
        "--data-dir",
        default=".agent_memory",
        help="Persistence directory (default: .agent_memory)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--backend",
            choices=["sqlite", "chromadb"],
            default="sqlite",
            help="Storage backend (default: sqlite)",
        )

    remember = sub.add_parser("remember", help="Store a memory")
    remember.add_argument("query")
    remember.add_argument("response")
    remember.add_argument("--type", default="conversation")
    remember.add_argument("--scope", default="user")
    remember.add_argument("--ttl", default=None)
    add_common_args(remember)
    remember.set_defaults(func=cmd_remember)

    resolve = sub.add_parser("resolve", help="Resolve a query against memory")
    resolve.add_argument("query")
    resolve.add_argument("--explain", action="store_true", help="Print score breakdown")
    add_common_args(resolve)
    resolve.set_defaults(func=cmd_resolve)

    stats = sub.add_parser("stats", help="Show memory statistics")
    add_common_args(stats)
    stats.set_defaults(func=cmd_stats)

    cleanup = sub.add_parser("cleanup", help="Expire or delete stale memories")
    cleanup.add_argument("--delete", action="store_true", help="Delete expired memories")
    add_common_args(cleanup)
    cleanup.set_defaults(func=cmd_cleanup)

    benchmark = sub.add_parser("benchmark", help="Run latency and hit-rate benchmark")
    benchmark.add_argument("queries", nargs="*", help="Queries to benchmark")
    benchmark.add_argument("--seed", action="store_true", help="Seed from eval datasets first")
    benchmark.add_argument("--repeat", type=int, default=1, help="Repeat each query N times")
    benchmark.add_argument("--baseline-ms", type=float, default=800.0, help="No-memory baseline ms")
    add_common_args(benchmark)
    benchmark.set_defaults(func=cmd_benchmark)

    eval_cmd = sub.add_parser("eval", help="Run evaluation datasets")
    eval_cmd.add_argument("--datasets", default=None, help="Path to datasets directory")
    add_common_args(eval_cmd)
    eval_cmd.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
