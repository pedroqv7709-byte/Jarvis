from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from jarvis.core.action import VerificationStatus
from jarvis.core.categories import ActionCategory
from jarvis.core.errors import ErrorKind, ToolError
from jarvis.tools.base import Tool, ToolArguments, ToolContext, ToolSpec
from jarvis.tools.builtin.filesystem import (
    DeleteFileTool,
    ListDirTool,
    OverwriteFileTool,
    ReadFileTool,
    WriteFileTool,
    filesystem_tools,
)
from jarvis.tools.registry import ToolRegistrationError, ToolRegistry
from jarvis.tools.workspace import Workspace

from support import CountingTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    ws = Workspace(tmp_path / "ws")
    ws.ensure()
    return ToolContext(workspace=ws, task_id="t", action_id="a1", attempt=1)


# -- registry -------------------------------------------------------------------------------


def test_registry_registers_and_describes_tools() -> None:
    registry = ToolRegistry(filesystem_tools())
    assert "fs.write_file" in registry
    assert registry.get("nope") is None
    catalog = {entry["name"]: entry for entry in registry.catalog()}
    schema = catalog["fs.write_file"]["input_schema"]
    assert schema["properties"]["path"]["type"] == "string"
    assert "path" in schema["required"]


def test_registry_rejects_duplicates() -> None:
    registry = ToolRegistry([CountingTool()])
    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.register(CountingTool())


def _tool_with(
    spec_updates: dict[str, Any], args_model: type[BaseModel] | None = None
) -> Tool[Any]:
    class Custom(CountingTool):
        spec = CountingTool.spec.model_copy(update=spec_updates)

    if args_model is not None:
        Custom.args_model = args_model  # type: ignore[assignment]
    return Custom()


def test_registry_rejects_tool_without_permissions() -> None:
    with pytest.raises(ToolRegistrationError, match="permissions"):
        ToolRegistry([_tool_with({"required_permissions": frozenset()})])


def test_registry_rejects_tool_without_categories() -> None:
    with pytest.raises(ToolRegistrationError, match="category"):
        ToolRegistry([_tool_with({"categories": frozenset()})])


def test_registry_rejects_permissive_argument_models() -> None:
    class Loose(ToolArguments):
        model_config: ClassVar = {"extra": "allow"}  # type: ignore[misc]
        text: str

    with pytest.raises(ToolRegistrationError, match="forbid"):
        ToolRegistry([_tool_with({}, Loose)])


def test_tool_spec_is_immutable() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError on frozen models
        CountingTool.spec.required_permissions = frozenset()  # type: ignore[misc]


def test_tool_names_are_validated() -> None:
    with pytest.raises(Exception):  # noqa: B017
        ToolSpec(
            name="Bad Name",
            description="x",
            required_permissions=frozenset({"x"}),
            categories=frozenset({ActionCategory.READ}),
            side_effects=False,
            idempotent=True,
        )


# -- argument validation --------------------------------------------------------------------


def test_argument_validation_errors_are_structured() -> None:
    tool = WriteFileTool()
    with pytest.raises(ToolError) as excinfo:
        tool.validate_arguments({"content": "x"})
    assert excinfo.value.failure.kind is ErrorKind.VALIDATION
    assert excinfo.value.failure.details["errors"][0]["loc"] == ("path",)


def test_unknown_arguments_are_rejected() -> None:
    with pytest.raises(ToolError) as excinfo:
        ReadFileTool().validate_arguments({"path": "a.txt", "sudo": True})
    assert excinfo.value.failure.kind is ErrorKind.VALIDATION


# -- workspace confinement ------------------------------------------------------------------


@pytest.mark.parametrize("path", ["../outside.txt", "a/../../outside.txt", "/etc/passwd", "C:/x"])
def test_workspace_rejects_escapes(ctx: ToolContext, path: str) -> None:
    with pytest.raises(ToolError) as excinfo:
        ctx.workspace.resolve(path)
    assert excinfo.value.failure.kind is ErrorKind.POLICY_DENIED
    assert excinfo.value.failure.details["sandbox_violation"] is True


def test_workspace_allows_nested_paths(ctx: ToolContext) -> None:
    resolved = ctx.workspace.resolve("sub/dir/file.txt")
    assert resolved.is_relative_to(ctx.workspace.root)


# -- filesystem tools -----------------------------------------------------------------------


def test_write_then_verify_then_read(ctx: ToolContext) -> None:
    write = WriteFileTool()
    args = write.validate_arguments({"path": "notes/a.txt", "content": "hola ñ"})
    out = write.run(args, ctx)
    assert out["path"] == "notes/a.txt"
    assert out["bytes"] == len("hola ñ".encode())
    assert write.verify(args, out, ctx).status is VerificationStatus.VERIFIED

    read = ReadFileTool()
    result = read.run(read.validate_arguments({"path": "notes/a.txt"}), ctx)
    assert result["content"] == "hola ñ"
    assert result["sha256"] == out["sha256"]


def test_write_never_overwrites(ctx: ToolContext) -> None:
    write = WriteFileTool()
    write.run(write.validate_arguments({"path": "a.txt", "content": "1"}), ctx)
    with pytest.raises(ToolError, match="already exists"):
        write.run(write.validate_arguments({"path": "a.txt", "content": "2"}), ctx)
    with pytest.raises(ToolError):
        write.validate_arguments({"path": "a.txt", "content": "2", "overwrite": True})
    assert (ctx.workspace.root / "a.txt").read_text(encoding="utf-8") == "1"


def test_overwrite_replaces_existing_files_only(ctx: ToolContext) -> None:
    tool = OverwriteFileTool()
    with pytest.raises(ToolError, match="not an existing file"):
        tool.run(tool.validate_arguments({"path": "new.txt", "content": "x"}), ctx)
    (ctx.workspace.root / "a.txt").write_text("1", encoding="utf-8")
    args = tool.validate_arguments({"path": "a.txt", "content": "2"})
    out = tool.run(args, ctx)
    assert (ctx.workspace.root / "a.txt").read_text(encoding="utf-8") == "2"
    assert out["previous_sha256"] != out["sha256"]
    assert tool.verify(args, out, ctx).status is VerificationStatus.VERIFIED
    assert ActionCategory.DELETE in tool.spec.categories


def test_write_verification_detects_tampering(ctx: ToolContext) -> None:
    write = WriteFileTool()
    args = write.validate_arguments({"path": "a.txt", "content": "original"})
    out = write.run(args, ctx)
    (ctx.workspace.root / "a.txt").write_text("tampered", encoding="utf-8")
    result = write.verify(args, out, ctx)
    assert result.status is VerificationStatus.FAILED
    assert "expected_sha256" in result.evidence


def test_read_missing_file_is_a_tool_error(ctx: ToolContext) -> None:
    read = ReadFileTool()
    with pytest.raises(ToolError) as excinfo:
        read.run(read.validate_arguments({"path": "missing.txt"}), ctx)
    assert excinfo.value.failure.kind is ErrorKind.TOOL_ERROR
    assert not excinfo.value.failure.retryable


def test_list_dir(ctx: ToolContext) -> None:
    (ctx.workspace.root / "b.txt").write_text("x", encoding="utf-8")
    (ctx.workspace.root / "a").mkdir()
    out = ListDirTool().run(ListDirTool().validate_arguments({}), ctx)
    assert out["entries"] == [{"name": "a", "type": "dir"}, {"name": "b.txt", "type": "file"}]


def test_delete_and_verify(ctx: ToolContext) -> None:
    (ctx.workspace.root / "a.txt").write_text("x", encoding="utf-8")
    tool = DeleteFileTool()
    args = tool.validate_arguments({"path": "a.txt"})
    out = tool.run(args, ctx)
    assert not (ctx.workspace.root / "a.txt").exists()
    assert tool.verify(args, out, ctx).status is VerificationStatus.VERIFIED


def test_read_tool_has_no_verifier() -> None:
    assert not ReadFileTool().has_verifier
    assert WriteFileTool().has_verifier
