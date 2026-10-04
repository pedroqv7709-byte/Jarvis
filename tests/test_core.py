from __future__ import annotations

import pytest

from jarvis.core.action import Action, ActionRecord, ActionResult, ActionStatus, AttemptStatus
from jarvis.core.decisions import Decision, most_restrictive
from jarvis.core.errors import ErrorKind, Failure, InvalidTransitionError
from jarvis.core.request import Request
from jarvis.core.task import ALLOWED_TRANSITIONS, Task, TaskState


def _task() -> Task:
    return Task.from_request(Request(text="haz algo", user_id="u1"))


def test_new_task_is_pending_with_ids() -> None:
    task = _task()
    assert task.state is TaskState.PENDING
    assert task.id.startswith("task_")
    assert task.request.id.startswith("req_")
    assert task.goal == "haz algo"
    assert task.version == 0
    assert task.actions == []


def test_valid_transitions_are_recorded() -> None:
    task = _task()
    task.transition(TaskState.RUNNING, "start")
    task.transition(TaskState.WAITING_PERMISSION, "needs ok")
    task.transition(TaskState.RUNNING, "approved")
    task.transition(TaskState.COMPLETED, "done")
    assert task.state is TaskState.COMPLETED
    assert task.is_terminal
    assert [(t.from_state, t.to_state) for t in task.transitions] == [
        (TaskState.PENDING, TaskState.RUNNING),
        (TaskState.RUNNING, TaskState.WAITING_PERMISSION),
        (TaskState.WAITING_PERMISSION, TaskState.RUNNING),
        (TaskState.RUNNING, TaskState.COMPLETED),
    ]


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (TaskState.PENDING, TaskState.COMPLETED),
        (TaskState.PENDING, TaskState.WAITING_PERMISSION),
        (TaskState.COMPLETED, TaskState.RUNNING),
        (TaskState.FAILED, TaskState.RUNNING),
        (TaskState.CANCELLED, TaskState.PENDING),
    ],
)
def test_invalid_transitions_raise(start: TaskState, target: TaskState) -> None:
    task = _task()
    task.state = start
    with pytest.raises(InvalidTransitionError):
        task.transition(target, "nope")
    assert task.state is start


def test_terminal_states_have_no_exits() -> None:
    for state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
        assert ALLOWED_TRANSITIONS[state] == frozenset()
        assert state.is_terminal


def test_task_round_trips_through_json() -> None:
    task = _task()
    record = ActionRecord(action=Action(id="a1", tool="fs.read_file", arguments={"path": "x"}))
    record.status = ActionStatus.FAILED
    record.attempts.append(
        ActionResult(
            action_id="a1",
            attempt=1,
            status=AttemptStatus.FAILED,
            failure=Failure(kind=ErrorKind.TRANSIENT, message="x", retryable=True),
        )
    )
    task.actions.append(record)
    task.transition(TaskState.RUNNING, "start")

    restored = Task.model_validate_json(task.model_dump_json())
    assert restored == task
    assert restored.actions[0].attempts[0].failure is not None
    assert restored.actions[0].attempts[0].failure.kind is ErrorKind.TRANSIENT


def test_action_digest_binds_tool_and_arguments() -> None:
    a = Action(id="a1", tool="fs.write_file", arguments={"path": "x", "content": "1"})
    same = Action(id="a9", tool="fs.write_file", arguments={"content": "1", "path": "x"})
    other = Action(id="a1", tool="fs.write_file", arguments={"path": "x", "content": "2"})
    assert a.digest() == same.digest()
    assert a.digest() != other.digest()


def test_most_restrictive_decision() -> None:
    assert most_restrictive(Decision.ALLOW, Decision.ALLOW) is Decision.ALLOW
    assert (
        most_restrictive(Decision.ALLOW, Decision.REQUIRE_CONFIRMATION)
        is Decision.REQUIRE_CONFIRMATION
    )
    assert most_restrictive(Decision.REQUIRE_CONFIRMATION, Decision.DENY) is Decision.DENY
    with pytest.raises(ValueError):
        most_restrictive()
