"""Filesystem tools confined to the workspace."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field

from jarvis.core.action import VerificationResult, VerificationStatus
from jarvis.core.categories import ActionCategory
from jarvis.core.errors import ErrorKind, ToolError
from jarvis.tools.base import Tool, ToolArguments, ToolContext, ToolSpec

MAX_FILE_BYTES = 1_000_000


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(target: Path, data: bytes) -> None:
    """Write to a temporary file and atomically replace, so a crash never leaves a partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".jarvis-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _verify_content(path: str, content: str, ctx: ToolContext) -> VerificationResult:
    target = ctx.workspace.resolve(path)
    if not target.is_file():
        return VerificationResult(
            status=VerificationStatus.FAILED, message=f"{path!r} does not exist"
        )
    actual = _sha256(target.read_bytes())
    expected = _sha256(content.encode("utf-8"))
    if actual != expected:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            message="file content does not match what was written",
            evidence={"expected_sha256": expected, "actual_sha256": actual},
        )
    return VerificationResult(
        status=VerificationStatus.VERIFIED,
        message="file exists with the expected content",
        evidence={"sha256": actual},
    )


class WriteFileArgs(ToolArguments):
    path: str = Field(min_length=1, max_length=255, description="Path relative to the workspace.")
    content: str = Field(default="", max_length=MAX_FILE_BYTES)


class WriteFileTool(Tool[WriteFileArgs]):
    spec = ToolSpec(
        name="fs.write_file",
        description="Create a new text file in the workspace. Fails if the file already exists.",
        required_permissions=frozenset({"fs.write"}),
        categories=frozenset({ActionCategory.WRITE}),
        side_effects=True,
        idempotent=False,
    )
    args_model = WriteFileArgs

    def run(self, args: WriteFileArgs, ctx: ToolContext) -> dict[str, Any]:
        target = ctx.workspace.resolve(args.path)
        if target.exists():
            raise ToolError(ErrorKind.TOOL_ERROR, f"{args.path!r} already exists")
        data = args.content.encode("utf-8")
        _atomic_write(target, data)
        return {"path": ctx.workspace.relative(target), "bytes": len(data), "sha256": _sha256(data)}

    def verify(
        self, args: WriteFileArgs, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult:
        return _verify_content(args.path, args.content, ctx)


class OverwriteFileTool(Tool[WriteFileArgs]):
    """Replaces an existing file. Destroys the previous content, hence the DELETE category."""

    spec = ToolSpec(
        name="fs.overwrite_file",
        description="Replace the content of an existing text file in the workspace.",
        required_permissions=frozenset({"fs.write"}),
        categories=frozenset({ActionCategory.WRITE, ActionCategory.DELETE}),
        side_effects=True,
        idempotent=True,
    )
    args_model = WriteFileArgs

    def run(self, args: WriteFileArgs, ctx: ToolContext) -> dict[str, Any]:
        target = ctx.workspace.resolve(args.path)
        if not target.is_file():
            raise ToolError(ErrorKind.TOOL_ERROR, f"{args.path!r} is not an existing file")
        previous = _sha256(target.read_bytes())
        data = args.content.encode("utf-8")
        _atomic_write(target, data)
        return {
            "path": ctx.workspace.relative(target),
            "bytes": len(data),
            "sha256": _sha256(data),
            "previous_sha256": previous,
        }

    def verify(
        self, args: WriteFileArgs, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult:
        return _verify_content(args.path, args.content, ctx)


class ReadFileArgs(ToolArguments):
    path: str = Field(min_length=1, max_length=255, description="Path relative to the workspace.")


class ReadFileTool(Tool[ReadFileArgs]):
    spec = ToolSpec(
        name="fs.read_file",
        description="Read a UTF-8 text file from the workspace.",
        required_permissions=frozenset({"fs.read"}),
        categories=frozenset({ActionCategory.READ}),
        side_effects=False,
        idempotent=True,
    )
    args_model = ReadFileArgs

    def run(self, args: ReadFileArgs, ctx: ToolContext) -> dict[str, Any]:
        target = ctx.workspace.resolve(args.path)
        if not target.is_file():
            raise ToolError(ErrorKind.TOOL_ERROR, f"{args.path!r} is not an existing file")
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ToolError(ErrorKind.TOOL_ERROR, f"{args.path!r} is too large ({size} bytes)")
        data = target.read_bytes()
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(ErrorKind.TOOL_ERROR, f"{args.path!r} is not UTF-8 text") from exc
        return {
            "path": ctx.workspace.relative(target),
            "content": content,
            "bytes": len(data),
            "sha256": _sha256(data),
        }


class ListDirArgs(ToolArguments):
    path: str = Field(default=".", min_length=1, max_length=255)


class ListDirTool(Tool[ListDirArgs]):
    spec = ToolSpec(
        name="fs.list_dir",
        description="List the entries of a directory in the workspace.",
        required_permissions=frozenset({"fs.read"}),
        categories=frozenset({ActionCategory.READ}),
        side_effects=False,
        idempotent=True,
    )
    args_model = ListDirArgs

    def run(self, args: ListDirArgs, ctx: ToolContext) -> dict[str, Any]:
        target = ctx.workspace.resolve(args.path)
        if not target.is_dir():
            raise ToolError(ErrorKind.TOOL_ERROR, f"{args.path!r} is not a directory")
        entries = [
            {"name": child.name, "type": "dir" if child.is_dir() else "file"}
            for child in sorted(target.iterdir(), key=lambda p: p.name)
            if not child.name.startswith(".jarvis-")
        ]
        return {"path": ctx.workspace.relative(target), "entries": entries}


class DeleteFileArgs(ToolArguments):
    path: str = Field(min_length=1, max_length=255, description="Path relative to the workspace.")


class DeleteFileTool(Tool[DeleteFileArgs]):
    spec = ToolSpec(
        name="fs.delete_file",
        description="Delete a file from the workspace.",
        required_permissions=frozenset({"fs.delete"}),
        categories=frozenset({ActionCategory.DELETE}),
        side_effects=True,
        idempotent=False,
    )
    args_model = DeleteFileArgs

    def run(self, args: DeleteFileArgs, ctx: ToolContext) -> dict[str, Any]:
        target = ctx.workspace.resolve(args.path)
        if not target.is_file():
            raise ToolError(ErrorKind.TOOL_ERROR, f"{args.path!r} is not an existing file")
        digest = _sha256(target.read_bytes())
        target.unlink()
        return {"path": ctx.workspace.relative(target), "deleted_sha256": digest}

    def verify(
        self, args: DeleteFileArgs, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult:
        if ctx.workspace.resolve(args.path).exists():
            return VerificationResult(
                status=VerificationStatus.FAILED, message=f"{args.path!r} still exists"
            )
        return VerificationResult(
            status=VerificationStatus.VERIFIED, message="file no longer exists"
        )


def filesystem_tools() -> list[Tool[Any]]:
    return [WriteFileTool(), OverwriteFileTool(), ReadFileTool(), ListDirTool(), DeleteFileTool()]
