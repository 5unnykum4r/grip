"""Tests for the workflow engine: models, template resolution, store, and execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from grip.engines.types import AgentRunResult, EngineProtocol
from grip.tools.base import ToolContext
from grip.tools.workflow import WorkflowTool
from grip.workflow.engine import MAX_TEMPLATE_OUTPUT_LENGTH, WorkflowEngine
from grip.workflow.models import StepDef, StepResult, StepStatus, WorkflowDef
from grip.workflow.store import WorkflowStore

# ===================================================================
# Model validation tests (existing)
# ===================================================================


def test_workflow_validation_passes():
    wf = WorkflowDef(
        name="test",
        steps=[
            StepDef(name="a", prompt="do A"),
            StepDef(name="b", prompt="do B", depends_on=["a"]),
        ],
    )
    assert wf.validate() == []


def test_workflow_validation_missing_dep():
    wf = WorkflowDef(
        name="test",
        steps=[
            StepDef(name="a", prompt="do A", depends_on=["nonexistent"]),
        ],
    )
    errors = wf.validate()
    assert len(errors) == 1
    assert "nonexistent" in errors[0]


def test_workflow_validation_cycle():
    wf = WorkflowDef(
        name="test",
        steps=[
            StepDef(name="a", prompt="A", depends_on=["b"]),
            StepDef(name="b", prompt="B", depends_on=["a"]),
        ],
    )
    errors = wf.validate()
    assert any("Circular" in e for e in errors)


def test_execution_order_parallel():
    wf = WorkflowDef(
        name="test",
        steps=[
            StepDef(name="a", prompt="A"),
            StepDef(name="b", prompt="B"),
            StepDef(name="c", prompt="C", depends_on=["a", "b"]),
        ],
    )
    layers = wf.get_execution_order()
    assert len(layers) == 2
    assert set(layers[0]) == {"a", "b"}
    assert layers[1] == ["c"]


def test_execution_order_sequential():
    wf = WorkflowDef(
        name="test",
        steps=[
            StepDef(name="a", prompt="A"),
            StepDef(name="b", prompt="B", depends_on=["a"]),
            StepDef(name="c", prompt="C", depends_on=["b"]),
        ],
    )
    layers = wf.get_execution_order()
    assert len(layers) == 3
    assert layers[0] == ["a"]
    assert layers[1] == ["b"]
    assert layers[2] == ["c"]


def test_step_result_lifecycle():
    result = StepResult(name="test_step")
    assert result.status == StepStatus.PENDING

    result.mark_running()
    assert result.status == StepStatus.RUNNING
    assert result.started_at is not None

    result.mark_completed("output text", iterations=3)
    assert result.status == StepStatus.COMPLETED
    assert result.output == "output text"
    assert result.iterations == 3


def test_step_result_failure():
    result = StepResult(name="fail_step")
    result.mark_running()
    result.mark_failed("something broke")
    assert result.status == StepStatus.FAILED
    assert result.error == "something broke"


def test_workflow_to_dict_roundtrip():
    wf = WorkflowDef(
        name="roundtrip",
        description="Test roundtrip",
        steps=[
            StepDef(name="s1", prompt="step 1", profile="researcher"),
            StepDef(name="s2", prompt="step 2 using {{s1.output}}", depends_on=["s1"]),
        ],
    )
    data = wf.to_dict()
    restored = WorkflowDef.from_dict(data)
    assert restored.name == "roundtrip"
    assert len(restored.steps) == 2
    assert restored.steps[1].depends_on == ["s1"]


# ===================================================================
# Template resolution tests
# ===================================================================


class TestTemplateResolution:
    def test_basic_substitution(self):
        results = {"step_a": StepResult(name="step_a")}
        results["step_a"].mark_running()
        results["step_a"].mark_completed("hello world", iterations=1)

        resolved = WorkflowEngine._resolve_template(
            "Use this: {{step_a.output}}", results
        )
        assert "hello world" in resolved
        assert "{{step_a.output}}" not in resolved

    def test_unresolved_left_intact(self):
        results = {"step_a": StepResult(name="step_a")}

        resolved = WorkflowEngine._resolve_template(
            "Use this: {{step_a.output}}", results
        )
        assert "{{step_a.output}}" in resolved

    def test_injection_stripped(self):
        """Output containing {{xxx.output}} patterns must NOT be re-expanded."""
        results = {
            "step_a": StepResult(name="step_a"),
            "step_b": StepResult(name="step_b"),
        }
        results["step_a"].mark_running()
        results["step_a"].mark_completed("secret: {{step_b.output}}", iterations=1)
        results["step_b"].mark_running()
        results["step_b"].mark_completed("LEAKED", iterations=1)

        resolved = WorkflowEngine._resolve_template(
            "Process: {{step_a.output}}", results
        )
        assert "LEAKED" not in resolved
        assert "[template-ref-removed]" in resolved

    def test_output_wrapped_in_delimiters(self):
        results = {"step_a": StepResult(name="step_a")}
        results["step_a"].mark_running()
        results["step_a"].mark_completed("some data", iterations=1)

        resolved = WorkflowEngine._resolve_template("{{step_a.output}}", results)
        assert "[output from step_a]" in resolved
        assert "[/output from step_a]" in resolved

    def test_output_truncated_at_limit(self):
        huge_output = "x" * (MAX_TEMPLATE_OUTPUT_LENGTH + 1000)
        results = {"step_a": StepResult(name="step_a")}
        results["step_a"].mark_running()
        results["step_a"].mark_completed(huge_output, iterations=1)

        resolved = WorkflowEngine._resolve_template("{{step_a.output}}", results)
        assert "[truncated]" in resolved
        assert len(resolved) < MAX_TEMPLATE_OUTPUT_LENGTH + 500


# ===================================================================
# Store CRUD tests
# ===================================================================


class TestWorkflowStore:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        store = WorkflowStore(tmp_path / "workflows")
        wf = WorkflowDef(
            name="test-wf",
            description="A test workflow",
            steps=[StepDef(name="s1", prompt="do something")],
        )
        store.save(wf)
        loaded = store.load("test-wf")
        assert loaded is not None
        assert loaded.name == "test-wf"
        assert len(loaded.steps) == 1
        assert loaded.steps[0].prompt == "do something"

    def test_list_workflows(self, tmp_path: Path):
        store = WorkflowStore(tmp_path / "workflows")
        store.save(WorkflowDef(name="alpha", steps=[StepDef(name="a", prompt="A")]))
        store.save(WorkflowDef(name="beta", steps=[StepDef(name="b", prompt="B")]))
        names = store.list_workflows()
        assert names == ["alpha", "beta"]

    def test_delete_existing(self, tmp_path: Path):
        store = WorkflowStore(tmp_path / "workflows")
        store.save(WorkflowDef(name="to-delete", steps=[]))
        assert store.delete("to-delete") is True
        assert store.load("to-delete") is None

    def test_delete_nonexistent(self, tmp_path: Path):
        store = WorkflowStore(tmp_path / "workflows")
        assert store.delete("nope") is False

    def test_path_traversal_rejected(self, tmp_path: Path):
        store = WorkflowStore(tmp_path / "workflows")
        with pytest.raises(ValueError, match="Invalid workflow name"):
            store.save(WorkflowDef(name="../escape", steps=[]))


# ===================================================================
# Engine execution tests (mocked EngineProtocol)
# ===================================================================


def _make_mock_engine(responses: dict[str, str]):
    """Create a mock EngineProtocol returning predetermined responses by step name."""
    mock = AsyncMock(spec=EngineProtocol)

    async def mock_run(user_message, *, session_key="", model=None):
        step_name = session_key.split(":")[-1] if ":" in session_key else session_key
        return AgentRunResult(
            response=responses.get(step_name, "default response"),
            iterations=1,
        )

    mock.run = mock_run
    return mock


def _make_config():
    from grip.config.schema import GripConfig

    return GripConfig()


def _make_registry():
    from grip.tools.base import ToolRegistry

    return ToolRegistry()


class TestWorkflowEngine:
    def test_simple_sequential_workflow(self):
        wf = WorkflowDef(
            name="seq-test",
            steps=[
                StepDef(name="first", prompt="do first"),
                StepDef(
                    name="second",
                    prompt="do second with {{first.output}}",
                    depends_on=["first"],
                ),
            ],
        )
        mock_engine = _make_mock_engine({"first": "result-A", "second": "result-B"})
        engine = WorkflowEngine(_make_config(), mock_engine, _make_registry())
        result = asyncio.run(engine.run(wf))
        assert result.status == "completed"
        assert result.step_results["first"].status == StepStatus.COMPLETED
        assert result.step_results["second"].status == StepStatus.COMPLETED

    def test_parallel_steps_both_run(self):
        wf = WorkflowDef(
            name="par-test",
            steps=[
                StepDef(name="a", prompt="A"),
                StepDef(name="b", prompt="B"),
            ],
        )
        mock_engine = _make_mock_engine({"a": "done-a", "b": "done-b"})
        engine = WorkflowEngine(_make_config(), mock_engine, _make_registry())
        result = asyncio.run(engine.run(wf))
        assert result.status == "completed"
        assert result.step_results["a"].output == "done-a"
        assert result.step_results["b"].output == "done-b"

    def test_failure_skips_dependents_but_not_independent(self):
        """When step 'a' fails, 'c' (depends on a) is SKIPPED, but
        'd' (depends on 'b' which succeeded) still runs."""
        wf = WorkflowDef(
            name="fail-test",
            steps=[
                StepDef(name="a", prompt="A"),
                StepDef(name="b", prompt="B"),
                StepDef(name="c", prompt="C from {{a.output}}", depends_on=["a"]),
                StepDef(name="d", prompt="D from {{b.output}}", depends_on=["b"]),
            ],
        )

        async def mock_run(user_message, *, session_key="", model=None):
            step_name = session_key.split(":")[-1]
            if step_name == "a":
                raise RuntimeError("step a exploded")
            return AgentRunResult(response=f"ok-{step_name}", iterations=1)

        mock_engine = AsyncMock(spec=EngineProtocol)
        mock_engine.run = mock_run

        engine = WorkflowEngine(_make_config(), mock_engine, _make_registry())
        result = asyncio.run(engine.run(wf))

        assert result.step_results["a"].status == StepStatus.FAILED
        assert result.step_results["b"].status == StepStatus.COMPLETED
        assert result.step_results["c"].status == StepStatus.SKIPPED
        assert result.step_results["d"].status == StepStatus.COMPLETED
        assert result.status == "failed"

    def test_timeout_marks_step_failed(self):
        wf = WorkflowDef(
            name="timeout-test",
            steps=[StepDef(name="slow", prompt="be slow", timeout_seconds=1)],
        )

        async def mock_run(user_message, *, session_key="", model=None):
            await asyncio.sleep(10)
            return AgentRunResult(response="never reached", iterations=1)

        mock_engine = AsyncMock(spec=EngineProtocol)
        mock_engine.run = mock_run

        engine = WorkflowEngine(_make_config(), mock_engine, _make_registry())
        result = asyncio.run(engine.run(wf))
        assert result.step_results["slow"].status == StepStatus.FAILED
        assert "Timed out" in result.step_results["slow"].error


# ===================================================================
# Metrics wiring test
# ===================================================================


def test_workflow_run_records_metrics(monkeypatch):
    """Verify that record_workflow_run() is called after engine.run()."""
    from grip.observe.metrics import MetricsCollector

    mock_collector = MetricsCollector()
    monkeypatch.setattr("grip.workflow.engine.get_metrics", lambda: mock_collector)

    wf = WorkflowDef(
        name="metrics-test",
        steps=[StepDef(name="s", prompt="go")],
    )
    mock_engine = _make_mock_engine({"s": "done"})
    engine = WorkflowEngine(_make_config(), mock_engine, _make_registry())
    asyncio.run(engine.run(wf))

    snap = mock_collector.snapshot()
    assert snap.total_workflow_runs == 1


# ===================================================================
# Additional model validation tests
# ===================================================================


def test_validation_empty_name():
    wf = WorkflowDef(name="", steps=[StepDef(name="a", prompt="do A")])
    errors = wf.validate()
    assert any("empty" in e.lower() for e in errors)


def test_validation_invalid_step_name():
    wf = WorkflowDef(name="test", steps=[StepDef(name="bad step!", prompt="go")])
    errors = wf.validate()
    assert any("invalid" in e.lower() for e in errors)


def test_validation_empty_prompt():
    wf = WorkflowDef(name="test", steps=[StepDef(name="a", prompt="")])
    errors = wf.validate()
    assert any("empty prompt" in e.lower() for e in errors)


def test_validation_bad_timeout():
    wf = WorkflowDef(name="test", steps=[StepDef(name="a", prompt="go", timeout_seconds=0)])
    errors = wf.validate()
    assert any("timeout" in e.lower() for e in errors)


def test_validation_duplicate_step_names():
    wf = WorkflowDef(
        name="test",
        steps=[
            StepDef(name="a", prompt="first"),
            StepDef(name="a", prompt="second"),
        ],
    )
    errors = wf.validate()
    assert any("duplicate" in e.lower() for e in errors)


def test_step_result_mark_skipped():
    result = StepResult(name="skip_step")
    result.mark_running()
    result.mark_skipped("dependency failed")
    assert result.status == StepStatus.SKIPPED
    assert result.error == "dependency failed"
    assert result.completed_at is not None


def test_hyphenated_step_name_in_template():
    results = {"code-review": StepResult(name="code-review")}
    results["code-review"].mark_running()
    results["code-review"].mark_completed("all looks good", iterations=1)

    resolved = WorkflowEngine._resolve_template(
        "Review says: {{code-review.output}}", results
    )
    assert "all looks good" in resolved
    assert "{{code-review.output}}" not in resolved


# ===================================================================
# Store edge-case tests
# ===================================================================


def test_store_load_invalid_name(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows")
    assert store.load("../bad") is None


def test_store_delete_invalid_name(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows")
    assert store.delete("../bad") is False


def test_store_load_nonexistent(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows")
    assert store.load("does-not-exist") is None


# ===================================================================
# WorkflowTool tests (agent-accessible CRUD)
# ===================================================================


class TestWorkflowTool:
    def _make_ctx(self, tmp_path: Path) -> ToolContext:
        return ToolContext(workspace_path=tmp_path)

    def test_create_workflow(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(
            tool.execute(
                {
                    "action": "create",
                    "workflow_name": "my-wf",
                    "description": "Test workflow",
                    "steps": [
                        {"name": "s1", "prompt": "do step 1"},
                        {"name": "s2", "prompt": "do step 2", "depends_on": ["s1"]},
                    ],
                },
                ctx,
            )
        )
        assert "created successfully" in result
        assert "my-wf" in result

    def test_create_duplicate_rejected(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        params = {
            "action": "create",
            "workflow_name": "dup",
            "steps": [{"name": "s1", "prompt": "go"}],
        }
        asyncio.run(tool.execute(params, ctx))
        result = asyncio.run(tool.execute(params, ctx))
        assert "already exists" in result

    def test_create_validation_error(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(
            tool.execute(
                {
                    "action": "create",
                    "workflow_name": "bad",
                    "steps": [{"name": "a", "prompt": "go", "depends_on": ["nonexistent"]}],
                },
                ctx,
            )
        )
        assert "validation failed" in result

    def test_list_empty(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(tool.execute({"action": "list"}, ctx))
        assert "No workflows found" in result

    def test_list_after_create(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        asyncio.run(
            tool.execute(
                {
                    "action": "create",
                    "workflow_name": "alpha",
                    "steps": [{"name": "s1", "prompt": "go"}],
                },
                ctx,
            )
        )
        result = asyncio.run(tool.execute({"action": "list"}, ctx))
        assert "alpha" in result

    def test_show_workflow(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        asyncio.run(
            tool.execute(
                {
                    "action": "create",
                    "workflow_name": "showme",
                    "steps": [
                        {"name": "a", "prompt": "do A"},
                        {"name": "b", "prompt": "do B", "depends_on": ["a"]},
                    ],
                },
                ctx,
            )
        )
        result = asyncio.run(
            tool.execute({"action": "show", "workflow_name": "showme"}, ctx)
        )
        assert "Workflow: showme" in result
        assert "Layer 1" in result
        assert "Layer 2" in result

    def test_show_nonexistent(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(
            tool.execute({"action": "show", "workflow_name": "nope"}, ctx)
        )
        assert "not found" in result

    def test_edit_workflow(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        asyncio.run(
            tool.execute(
                {
                    "action": "create",
                    "workflow_name": "editable",
                    "steps": [{"name": "s1", "prompt": "v1"}],
                },
                ctx,
            )
        )
        result = asyncio.run(
            tool.execute(
                {
                    "action": "edit",
                    "workflow_name": "editable",
                    "steps": [
                        {"name": "s1", "prompt": "v2"},
                        {"name": "s2", "prompt": "new step", "depends_on": ["s1"]},
                    ],
                },
                ctx,
            )
        )
        assert "updated successfully" in result
        show = asyncio.run(
            tool.execute({"action": "show", "workflow_name": "editable"}, ctx)
        )
        assert "s2" in show

    def test_edit_nonexistent_rejected(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(
            tool.execute(
                {
                    "action": "edit",
                    "workflow_name": "nope",
                    "steps": [{"name": "s1", "prompt": "go"}],
                },
                ctx,
            )
        )
        assert "not found" in result

    def test_delete_workflow(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        asyncio.run(
            tool.execute(
                {
                    "action": "create",
                    "workflow_name": "removeme",
                    "steps": [{"name": "s1", "prompt": "go"}],
                },
                ctx,
            )
        )
        result = asyncio.run(
            tool.execute({"action": "delete", "workflow_name": "removeme"}, ctx)
        )
        assert "deleted" in result.lower()
        result2 = asyncio.run(
            tool.execute({"action": "show", "workflow_name": "removeme"}, ctx)
        )
        assert "not found" in result2

    def test_delete_nonexistent(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(
            tool.execute({"action": "delete", "workflow_name": "nope"}, ctx)
        )
        assert "not found" in result

    def test_unknown_action(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(tool.execute({"action": "bogus"}, ctx))
        assert "unknown action" in result.lower()

    def test_create_missing_name(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(
            tool.execute(
                {"action": "create", "steps": [{"name": "s1", "prompt": "go"}]},
                ctx,
            )
        )
        assert "workflow_name is required" in result

    def test_create_missing_steps(self, tmp_path: Path):
        tool = WorkflowTool()
        ctx = self._make_ctx(tmp_path)
        result = asyncio.run(
            tool.execute({"action": "create", "workflow_name": "nostepper"}, ctx)
        )
        assert "steps array is required" in result
