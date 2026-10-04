from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from jarvis.core.action import Action
from jarvis.core.errors import PlanningError
from jarvis.models.base import ChatMessage, ChatResponse, ModelCapabilities, ModelError
from jarvis.models.router import ModelRouter, NoModelAvailableError
from jarvis.planning.base import Plan, PlanningRequest
from jarvis.planning.model_planner import ModelPlanner
from jarvis.planning.rule_based import RuleBasedPlanner, split_clauses
from jarvis.planning.validation import PlanValidator
from jarvis.tools.builtin.filesystem import filesystem_tools
from jarvis.tools.registry import ToolRegistry

from support import make_context


def _plan(goal: str) -> Plan:
    return RuleBasedPlanner().plan(PlanningRequest(goal=goal, context=make_context()))


def _single(goal: str) -> Action:
    plan = _plan(goal)
    assert len(plan.actions) == 1
    return plan.actions[0]


# -- rule-based planner ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("goal", "tool", "arguments"),
    [
        (
            "crea un archivo de prueba",
            "fs.write_file",
            {"path": "prueba.txt", "content": "Archivo de prueba creado por Jarvis.\n"},
        ),
        (
            "create a test file",
            "fs.write_file",
            {"path": "test.txt", "content": "Archivo de prueba creado por Jarvis.\n"},
        ),
        (
            'crea el archivo notas.txt con el contenido "hola mundo"',
            "fs.write_file",
            {"path": "notas.txt", "content": "hola mundo"},
        ),
        (
            "Jarvis, por favor crea el archivo a.txt.",
            "fs.write_file",
            {"path": "a.txt", "content": ""},
        ),
        (
            "write file docs/x.md with content # Title",
            "fs.write_file",
            {"path": "docs/x.md", "content": "# Title"},
        ),
        (
            "sobrescribe el archivo a.txt con el contenido nuevo",
            "fs.overwrite_file",
            {"path": "a.txt", "content": "nuevo"},
        ),
        ("lee el archivo notas.txt", "fs.read_file", {"path": "notas.txt"}),
        ("muestra el contenido del archivo a.txt", "fs.read_file", {"path": "a.txt"}),
        ("lista los archivos", "fs.list_dir", {"path": "."}),
        ("list files in docs", "fs.list_dir", {"path": "docs"}),
        ("borra el archivo notas.txt", "fs.delete_file", {"path": "notas.txt"}),
        ('delete file "my notes.txt"', "fs.delete_file", {"path": "my notes.txt"}),
    ],
)
def test_rule_based_intents(goal: str, tool: str, arguments: dict[str, Any]) -> None:
    action = _single(goal)
    assert action.tool == tool
    assert action.arguments == arguments


def test_multi_step_plan_with_dependencies_and_pronouns() -> None:
    plan = _plan("crea el archivo a.txt con contenido hola y luego léelo y después bórralo")
    assert [a.tool for a in plan.actions] == ["fs.write_file", "fs.read_file", "fs.delete_file"]
    assert [a.depends_on for a in plan.actions] == [(), ("a1",), ("a2",)]
    assert all(a.arguments["path"] == "a.txt" for a in plan.actions)


def test_quoted_text_is_never_split() -> None:
    assert split_clauses(
        'crea el archivo a.txt con contenido "x; y luego z"; lista los archivos'
    ) == [
        'crea el archivo a.txt con contenido "x; y luego z"',
        "lista los archivos",
    ]


@pytest.mark.parametrize("goal", ["haz magia", "crea un archivo", "léelo", "ciérrala", "   "])
def test_unsupported_goals_fail_loudly(goal: str) -> None:
    with pytest.raises(PlanningError):
        _plan(goal)


# -- plan validation ------------------------------------------------------------------------


@pytest.fixture
def validator() -> PlanValidator:
    return PlanValidator(ToolRegistry(filesystem_tools()), max_steps=3)


def _a(
    action_id: str, tool: str = "fs.list_dir", depends_on: tuple[str, ...] = (), **args: Any
) -> Action:
    return Action(id=action_id, tool=tool, arguments=args, depends_on=depends_on)


def test_valid_plan_passes(validator: PlanValidator) -> None:
    validator.validate(Plan(planner="t", actions=(_a("a1"), _a("a2", depends_on=("a1",)))))


@pytest.mark.parametrize(
    ("plan", "match"),
    [
        (Plan(planner="t"), "no actions"),
        (Plan(planner="t", actions=tuple(_a(f"a{i}") for i in range(4))), "limit"),
        (Plan(planner="t", actions=(_a("a1"), _a("a1"))), "duplicate"),
        (Plan(planner="t", actions=(_a("a1", depends_on=("a2",)), _a("a2"))), "earlier"),
        (Plan(planner="t", actions=(_a("a1", depends_on=("a1",)),)), "earlier"),
        (Plan(planner="t", actions=(_a("a1", tool="shell.exec"),)), "unknown tool"),
        (Plan(planner="t", actions=(_a("a1", tool="fs.read_file"),)), "invalid arguments"),
    ],
)
def test_invalid_plans_are_rejected(validator: PlanValidator, plan: Plan, match: str) -> None:
    with pytest.raises(PlanningError, match=match):
        validator.validate(plan)


def test_reply_only_plan_is_valid(validator: PlanValidator) -> None:
    validator.validate(Plan(planner="t", reply="hola"))


# -- model abstraction ----------------------------------------------------------------------


class ScriptedModel:
    """Test double implementing the ModelProvider protocol with canned structured output."""

    def __init__(self, name: str, payload: dict[str, Any] | Exception, **caps: Any) -> None:
        self.name = name
        self._payload = payload
        self._caps = ModelCapabilities(structured_output=True, **caps)
        self.seen: list[Sequence[ChatMessage]] = []

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def chat(self, messages: Sequence[ChatMessage], **options: Any) -> ChatResponse:
        return ChatResponse(content="", model=self.name)

    def structured_output[T: BaseModel](
        self, messages: Sequence[ChatMessage], schema: type[T], **options: Any
    ) -> T:
        self.seen.append(messages)
        if isinstance(self._payload, Exception):
            raise self._payload
        return schema.model_validate(self._payload)


def _model_plan(model: ScriptedModel) -> Plan:
    router = ModelRouter()
    router.register(model)
    registry = ToolRegistry(filesystem_tools())
    return ModelPlanner(router).plan(
        PlanningRequest(goal="haz algo", context=make_context(), catalog=registry.catalog())
    )


def test_model_planner_converts_proposals_into_actions() -> None:
    model = ScriptedModel(
        "m",
        {
            "actions": [
                {"tool": "fs.write_file", "arguments": {"path": "a.txt", "content": "x"}},
                {"tool": "fs.read_file", "arguments": {"path": "a.txt"}, "depends_on": [0]},
            ]
        },
    )
    plan = _model_plan(model)
    assert plan.planner == "model"
    assert [a.id for a in plan.actions] == ["a1", "a2"]
    assert plan.actions[1].depends_on == ("a1",)
    system_prompt = model.seen[0][0].content
    assert "fs.write_file" in system_prompt


def test_model_planner_never_sends_sensitive_context() -> None:
    from jarvis.context.model import ContextItem

    model = ScriptedModel("m", {"reply": "ok"})
    router = ModelRouter()
    router.register(model)
    context = make_context().model_copy(
        update={
            "items": (
                ContextItem(source="t", key="city", value="Madrid"),
                ContextItem(source="t", key="iban", value="ES00-SECRET", sensitive=True),
            )
        }
    )
    ModelPlanner(router).plan(PlanningRequest(goal="x", context=context))
    prompt = " ".join(m.content for m in model.seen[0])  # system + context payload
    assert "Madrid" in prompt
    assert "ES00-SECRET" not in prompt


def test_model_planner_reply_only() -> None:
    plan = _model_plan(ScriptedModel("m", {"reply": "Hola, ¿en qué te ayudo?"}))
    assert plan.actions == ()
    assert plan.reply == "Hola, ¿en qué te ayudo?"


@pytest.mark.parametrize(
    "payload",
    [
        {"actions": [{"tool": "fs.read_file", "arguments": {}, "depends_on": [0]}]},
        {"actions": [{"tool": "x", "unexpected": 1}]},
        ModelError("provider down"),
    ],
)
def test_model_planner_rejects_bad_output(payload: dict[str, Any] | Exception) -> None:
    with pytest.raises(PlanningError):
        _model_plan(ScriptedModel("m", payload))


def test_router_selects_by_capability_and_priority() -> None:
    router = ModelRouter()
    small = ScriptedModel("small", {}, local=True)
    big = ScriptedModel("big", {}, vision=True, max_context_tokens=200_000)
    router.register(small, priority=1)
    router.register(big)
    assert router.select().name == "small"
    assert router.select(ModelCapabilities(vision=True)).name == "big"
    assert router.select(ModelCapabilities(max_context_tokens=100_000)).name == "big"
    with pytest.raises(NoModelAvailableError):
        router.select(ModelCapabilities(audio=True))
    with pytest.raises(ValueError):
        router.register(ScriptedModel("big", {}))


def test_router_without_providers_fails_planning() -> None:
    planner = ModelPlanner(ModelRouter())
    with pytest.raises(PlanningError):
        planner.plan(PlanningRequest(goal="x", context=make_context()))
