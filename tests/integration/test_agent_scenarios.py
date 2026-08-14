# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


def test_daily_vague_accomplishment_clarification() -> None:
    """Tests that vague daily accomplishments prompt a short clarification question."""
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="today I fixed some bugs and coded")],
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )

    # Combine text responses from the stream
    response_text = "".join(
        part.text
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if part.text
    )

    # Check that it asks a clarification question
    assert len(response_text) > 0
    assert "?" in response_text or "Which" in response_text or "What" in response_text

    # Retrieve updated session state
    session_updated = session_service.get_session_sync(
        app_name="test", user_id="test_user", session_id=session.id
    )
    state = session_updated.state

    # It should have resolved and locked the target daily date
    assert state.get("target_daily_date") == datetime.now().strftime("%Y-%m-%d")


def test_daily_specific_accomplishment_approval() -> None:
    """Tests the daily logging flow with a specific accomplishment and approval."""
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    # Turn 1: Log specific accomplishment
    message1 = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="I fixed a bug in the query renderer that crashed the app."
            )
        ],
    )
    events1 = list(
        runner.run(
            new_message=message1,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    response1_text = "".join(
        part.text
        for event in events1
        if event.content and event.content.parts
        for part in event.content.parts
        if part.text
    )

    # Turn 1 should show the draft and ask if it looks good
    assert "renderer" in response1_text or "draft" in response1_text.lower()

    session_updated = session_service.get_session_sync(
        app_name="test", user_id="test_user", session_id=session.id
    )
    assert session_updated.state.get("target_daily_date") == datetime.now().strftime(
        "%Y-%m-%d"
    )
    assert "renderer" in session_updated.state.get("daily_draft", "").lower()

    # Turn 2: Approve the draft
    message2 = types.Content(
        role="user", parts=[types.Part.from_text(text="yes, save it")]
    )
    events2 = list(
        runner.run(
            new_message=message2,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )

    response2_text = "".join(
        part.text
        for event in events2
        if event.content and event.content.parts
        for part in event.content.parts
        if part.text
    )
    print("\n--- TURN 2 RESPONSE ---")
    print(response2_text)

    session_final = session_service.get_session_sync(
        app_name="test", user_id="test_user", session_id=session.id
    )
    print("\n--- TURN 2 STATE ---")
    print(session_final.state)
    # Assert that the response reports successful saving
    assert "saved" in response2_text.lower() or "approved" in response2_text.lower()
    # The active flow and draft variables should be reset to empty strings
    assert session_final.state.get("target_daily_date") == ""
    assert session_final.state.get("daily_draft") == ""
    assert session_final.state.get("active_flow") == ""
