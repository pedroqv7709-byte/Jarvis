from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.core.action import Action
from jarvis.core.categories import ActionCategory
from jarvis.core.decisions import Decision, RuleOutcome
from jarvis.permissions.model import GrantMode, PermissionSet
from jarvis.safety.policy import CategoryFloorRule, PolicyEngine, PolicyInput
from jarvis.tools.base import ToolSpec
from jarvis.tools.builtin.filesystem import DeleteFileTool, ReadFileTool, WriteFileTool

from support import CountingTool, DangerousTool, SendMessageTool, make_context


def _evaluate(
    spec: ToolSpec, grants: dict[str, str], engine: PolicyEngine | None = None
) -> Decision:
    engine = engine or PolicyEngine.default()
    action = Action(id="a1", tool=spec.name, arguments={})
    verdict = engine.evaluate(PolicyInput(action=action, spec=spec, context=make_context(grants)))
    return verdict.decision


# -- permission sets ------------------------------------------------------------------------


def test_permission_lookup_exact_wildcard_and_default_deny() -> None:
    perms = PermissionSet.of({"fs.*": "confirm", "fs.read": "allow", "*": "deny"})
    assert perms.mode_for("fs.read") is GrantMode.ALLOW
    assert perms.mode_for("fs.write") is GrantMode.CONFIRM
    assert perms.mode_for("messages.send") is GrantMode.DENY
    assert PermissionSet().mode_for("fs.read") is GrantMode.DENY


def test_longest_wildcard_wins() -> None:
    perms = PermissionSet.of({"a.*": "deny", "a.b.*": "allow"})
    assert perms.mode_for("a.b.c") is GrantMode.ALLOW
    assert perms.mode_for("a.x") is GrantMode.DENY


@pytest.mark.parametrize("scope", ["", "FS.read", "fs.*.x", "f*s", "fs read"])
def test_invalid_scopes_are_rejected(scope: str) -> None:
    with pytest.raises(ValidationError):
        PermissionSet.of({scope: "allow"})


# -- policy decisions -----------------------------------------------------------------------


def test_read_allowed_when_granted() -> None:
    assert _evaluate(ReadFileTool.spec, {"fs.read": "allow"}) is Decision.ALLOW


def test_denied_when_permission_not_granted() -> None:
    assert _evaluate(WriteFileTool.spec, {"fs.read": "allow"}) is Decision.DENY


def test_denied_when_grant_is_deny() -> None:
    assert _evaluate(WriteFileTool.spec, {"fs.write": "deny"}) is Decision.DENY


def test_confirm_grant_requires_confirmation() -> None:
    assert _evaluate(WriteFileTool.spec, {"fs.write": "confirm"}) is Decision.REQUIRE_CONFIRMATION


@pytest.mark.parametrize(
    "category",
    [
        ActionCategory.SEND_MESSAGE,
        ActionCategory.PURCHASE,
        ActionCategory.TRANSFER_MONEY,
        ActionCategory.CHANGE_CREDENTIALS,
        ActionCategory.DELETE,
        ActionCategory.SYSTEM_CONFIGURATION,
    ],
)
def test_sensitive_categories_require_confirmation_even_if_allowed(
    category: ActionCategory,
) -> None:
    spec = SendMessageTool.spec.model_copy(update={"categories": frozenset({category})})
    assert _evaluate(spec, {"messages.send": "allow"}) is Decision.REQUIRE_CONFIRMATION


def test_dangerous_is_denied_even_if_allowed() -> None:
    assert _evaluate(DangerousTool.spec, {"test.echo": "allow", "*": "allow"}) is Decision.DENY


def test_delete_tool_requires_confirmation_with_allow_grant() -> None:
    assert _evaluate(DeleteFileTool.spec, {"fs.delete": "allow"}) is Decision.REQUIRE_CONFIRMATION


def test_floor_overrides_can_only_tighten() -> None:
    engine = PolicyEngine.default(
        {ActionCategory.DANGEROUS: Decision.ALLOW, ActionCategory.READ: Decision.DENY}
    )
    assert _evaluate(DangerousTool.spec, {"test.echo": "allow"}, engine) is Decision.DENY
    assert _evaluate(CountingTool.spec, {"test.echo": "allow"}, engine) is Decision.DENY


def test_strictest_of_several_permissions_wins() -> None:
    spec = ReadFileTool.spec.model_copy(
        update={"required_permissions": frozenset({"fs.read", "net.access"})}
    )
    assert _evaluate(spec, {"fs.read": "allow", "net.access": "confirm"}) is (
        Decision.REQUIRE_CONFIRMATION
    )
    assert _evaluate(spec, {"fs.read": "allow"}) is Decision.DENY


def test_tool_without_permissions_is_denied() -> None:
    spec = ReadFileTool.spec.model_copy(update={"required_permissions": frozenset()})
    assert _evaluate(spec, {"*": "allow"}) is Decision.DENY


class _ExplodingRule:
    name = "exploding"

    def evaluate(self, inp: PolicyInput) -> RuleOutcome:
        raise RuntimeError("boom")


class _AbstainingRule:
    name = "abstain"

    def evaluate(self, inp: PolicyInput) -> None:
        return None


def test_rule_exception_fails_closed() -> None:
    from jarvis.safety.policy import PermissionGrantRule

    engine = PolicyEngine([PermissionGrantRule(), _ExplodingRule()])
    assert _evaluate(ReadFileTool.spec, {"fs.read": "allow"}, engine) is Decision.DENY


def test_all_rules_abstaining_means_deny() -> None:
    engine = PolicyEngine([_AbstainingRule(), CategoryFloorRule()])
    assert _evaluate(ReadFileTool.spec, {"fs.read": "allow"}, engine) is Decision.DENY


def test_verdict_explains_itself() -> None:
    engine = PolicyEngine.default()
    action = Action(id="a1", tool="fs.delete_file", arguments={"path": "x"})
    verdict = engine.evaluate(
        PolicyInput(
            action=action, spec=DeleteFileTool.spec, context=make_context({"fs.delete": "allow"})
        )
    )
    assert verdict.evaluated_permissions == ("fs.delete",)
    assert verdict.evaluated_categories == ("delete",)
    assert any("category_floor" in r for r in verdict.reasons)
    assert any("fs.delete=allow" in r for r in verdict.reasons)
