"""Regression tests for BUG-CUSTOM-FIELD-CLOBBER (47090242).

Root cause: stale UI reads send {field: null} for fields set by other actors;
the backend was deleting rows for null values, erasing operator-entered
provenance (branch, pr_url, test_evidence).

Fix: null in PATCH custom_field_values = "don't touch" (merge by omission).
Only explicit non-null values are written; null = preserve existing.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel, col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import tasks as tasks_api
from app.api.deps import ActorContext
from app.models.agents import Agent
from app.models.boards import Board
from app.models.gateways import Gateway
from app.models.organizations import Organization
from app.models.task_custom_fields import (
    BoardTaskCustomField,
    TaskCustomFieldDefinition,
    TaskCustomFieldValue,
)
from app.models.tasks import Task
from app.schemas.tasks import TaskUpdate


async def _make_engine() -> AsyncEngine:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.connect() as conn, conn.begin():
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


async def _make_session(engine: AsyncEngine) -> AsyncSession:
    return AsyncSession(engine, expire_on_commit=False)


async def _setup_board_with_custom_fields(
    session: AsyncSession,
) -> tuple[uuid4, uuid4, uuid4, uuid4]:
    """Create org, board, agent, task + 'branch' custom field definition.

    Returns (board_id, task_id, agent_id, branch_definition_id).
    """
    org_id = uuid4()
    board_id = uuid4()
    gateway_id = uuid4()
    agent_id = uuid4()
    task_id = uuid4()
    branch_def_id = uuid4()

    session.add(Organization(id=org_id, name="org"))
    session.add(
        Gateway(
            id=gateway_id,
            organization_id=org_id,
            name="gw",
            url="https://gw.local",
            workspace_root="/tmp",
        ),
    )
    session.add(
        Board(
            id=board_id,
            organization_id=org_id,
            name="board",
            slug="board",
            gateway_id=gateway_id,
        ),
    )
    session.add(
        Agent(
            id=agent_id,
            name="worker",
            board_id=board_id,
            gateway_id=gateway_id,
            status="online",
            is_board_lead=False,
        ),
    )
    session.add(
        Task(
            id=task_id,
            board_id=board_id,
            title="test task",
            status="in_progress",
            assigned_agent_id=agent_id,
        ),
    )
    # Define 'branch' custom field for the org
    session.add(
        TaskCustomFieldDefinition(
            id=branch_def_id,
            organization_id=org_id,
            field_key="branch",
            label="Branch",
            field_type="text",
        ),
    )
    # Bind it to the board
    session.add(
        BoardTaskCustomField(
            id=uuid4(),
            board_id=board_id,
            task_custom_field_definition_id=branch_def_id,
        ),
    )
    await session.commit()
    return board_id, task_id, agent_id, branch_def_id


@pytest.mark.asyncio
async def test_null_in_patch_does_not_clobber_existing_branch() -> None:
    """BUG-CUSTOM-FIELD-CLOBBER: {branch: null} in PATCH must not delete an existing branch row.

    Scenario: agent sets branch → stale UI sends PATCH with {branch: null} →
    branch must survive (null = "don't touch", not "clear").
    """
    engine = await _make_engine()
    try:
        async with await _make_session(engine) as session:
            board_id, task_id, agent_id, branch_def_id = await _setup_board_with_custom_fields(
                session,
            )

            # Step 1: agent writes branch value directly into the row
            session.add(
                TaskCustomFieldValue(
                    id=uuid4(),
                    task_id=task_id,
                    task_custom_field_definition_id=branch_def_id,
                    value="feat/my-branch",
                ),
            )
            await session.commit()

            # Verify row exists
            row = (
                await session.exec(
                    select(TaskCustomFieldValue).where(
                        col(TaskCustomFieldValue.task_id) == task_id,
                        col(TaskCustomFieldValue.task_custom_field_definition_id)
                        == branch_def_id,
                    ),
                )
            ).first()
            assert row is not None
            assert row.value == "feat/my-branch"

            # Step 2: stale UI PATCH with {branch: null} — simulates full payload from
            # an operator who loaded the task before the agent set branch
            task = (
                await session.exec(select(Task).where(col(Task.id) == task_id))
            ).first()
            assert task is not None

            agent = (
                await session.exec(select(Agent).where(col(Agent.id) == agent_id))
            ).first()
            assert agent is not None

            actor = ActorContext(actor_type="agent", agent=agent)
            payload = TaskUpdate(
                status="in_progress",
                custom_field_values={"branch": None},  # stale UI: null = "didn't see the value"
            )

            result = await tasks_api.update_task(
                payload=payload,
                task=task,
                session=session,
                actor=actor,
            )

            # Step 3: branch must survive
            assert result.custom_field_values is not None, "custom_field_values must be returned"
            assert result.custom_field_values.get("branch") == "feat/my-branch", (
                "null in PATCH must not clobber an existing branch value (BUG-CUSTOM-FIELD-CLOBBER)"
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_non_null_value_still_updates_branch() -> None:
    """Sanity: setting branch to a real value in PATCH still writes correctly."""
    engine = await _make_engine()
    try:
        async with await _make_session(engine) as session:
            board_id, task_id, agent_id, branch_def_id = await _setup_board_with_custom_fields(
                session,
            )

            task = (
                await session.exec(select(Task).where(col(Task.id) == task_id))
            ).first()
            assert task is not None
            agent = (
                await session.exec(select(Agent).where(col(Agent.id) == agent_id))
            ).first()
            assert agent is not None

            actor = ActorContext(actor_type="agent", agent=agent)
            payload = TaskUpdate(
                status="in_progress",
                custom_field_values={"branch": "feat/new-feature"},
            )

            result = await tasks_api.update_task(
                payload=payload,
                task=task,
                session=session,
                actor=actor,
            )

            assert result.custom_field_values is not None
            assert result.custom_field_values.get("branch") == "feat/new-feature"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_omitted_custom_field_values_key_is_no_op() -> None:
    """Sanity: PATCH without custom_field_values key at all must not touch existing values."""
    engine = await _make_engine()
    try:
        async with await _make_session(engine) as session:
            board_id, task_id, agent_id, branch_def_id = await _setup_board_with_custom_fields(
                session,
            )

            # Pre-set a branch value
            session.add(
                TaskCustomFieldValue(
                    id=uuid4(),
                    task_id=task_id,
                    task_custom_field_definition_id=branch_def_id,
                    value="feat/existing",
                ),
            )
            await session.commit()

            task = (
                await session.exec(select(Task).where(col(Task.id) == task_id))
            ).first()
            assert task is not None
            agent = (
                await session.exec(select(Agent).where(col(Agent.id) == agent_id))
            ).first()
            assert agent is not None

            actor = ActorContext(actor_type="agent", agent=agent)
            # No custom_field_values key in payload → custom_field_values_set=False → no-op
            payload = TaskUpdate(status="in_progress")

            result = await tasks_api.update_task(
                payload=payload,
                task=task,
                session=session,
                actor=actor,
            )

            assert result.custom_field_values is not None
            assert result.custom_field_values.get("branch") == "feat/existing"
    finally:
        await engine.dispose()
