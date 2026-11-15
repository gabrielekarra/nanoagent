"""Issue #7: the benchmark."""

import os
from pathlib import Path

from nanoagent import Model, Reply, bench
from nanoagent.testing import FakeModel, call, reply, text

DATA = Path(__file__).parent / "data"


def test_bench_has_twenty_tasks_with_solutions():
    assert len(bench.TASKS) == 20
    assert len({task.name for task in bench.TASKS}) == 20
    assert all(task.solution for task in bench.TASKS)


def test_bench_passes_with_perfect_script():
    results = bench.run(FakeModel(*bench.solutions()))
    failed = [(r.task, r.error) for r in results if not r.passed]
    assert failed == []
    assert all(r.steps >= 1 for r in results)


def test_bench_fails_wrong_outcome():
    task = next(t for t in bench.TASKS if t.name == "sum_numbers")
    (result,) = bench.run(FakeModel(reply(text("Done, the total is in total.txt."))), [task])
    assert not result.passed and result.steps == 1


def test_bench_step_limit():
    class Looping(Model):
        def reply(self, system, tools, messages, on_text=None):
            return Reply(calls=[__import__("nanoagent").Call(f"c{len(messages)}", "read_file", {"path": "notes.txt"})], stop="tool")

    task = next(t for t in bench.TASKS if t.name == "read_and_answer")
    (result,) = bench.run(Looping(), [task], step_limit=5)
    assert not result.passed and result.steps == 5 and "stopped after 5 steps" in result.error


def test_bench_table_format():
    results = [bench.Result("write_greeting", True, 2, 310, 1.2), bench.Result("sum_numbers", False, 4, 820, 3.4),
               bench.Result("chain_lookup", False, 12, 2100, 9.0, "stopped after 12 steps")]
    assert bench.table(results, "ollama:qwen3:4b") == (DATA / "bench_table.md").read_text().rstrip("\n")


def test_bench_tasks_are_isolated():
    here = os.getcwd()
    first = bench.Task("leave_file", "x", lambda folder, answer, calls: True,
                       solution=[], tools=bench.FILE_TOOLS)
    second = bench.Task("look_for_it", "y", lambda folder, answer, calls: not (folder / "left.txt").exists(), tools=bench.FILE_TOOLS)
    model = FakeModel(reply(call("write_file", path="left.txt", content="hi")), reply(text("ok")),
                      reply(call("read_file", path="left.txt")), reply(text("not there")))
    results = bench.run(model, [first, second])
    assert [r.passed for r in results] == [True, True]
    assert "does not exist" in model.tool_results(3)[0]["content"] or model.tool_results(3)[0]["error"]
    assert os.getcwd() == here
