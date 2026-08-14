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

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel, Field

# Import custom tools
from app import tools

load_dotenv()


# Pydantic input schemas to validate daily logging tool arguments
class StoreTargetLoggingDateInput(BaseModel):
    date: str = Field(
        description="The target daily date to log accomplishments for, in YYYY-MM-DD format."
    )


class SaveDailyAccomplishmentsDraftInput(BaseModel):
    text: str = Field(
        description="The bulleted list text of accomplishments draft to save to session state."
    )


class ApproveAndSaveDailyAccomplishmentsInput(BaseModel):
    text: str = Field(
        description="The exact verbatim text of approved daily accomplishments bulleted list."
    )
    date: str = Field(
        description="The target date in YYYY-MM-DD format for which achievements are logged."
    )


# Pydantic input schemas to validate weekly summary tool arguments
class StoreTargetWeeklySummaryMondayInput(BaseModel):
    monday_date: str = Field(
        description="The Monday date of the target week in YYYY-MM-DD format."
    )


class SaveWeeklySummaryDraftInput(BaseModel):
    draft: str = Field(
        description="The drafted markdown text of the weekly summary to save to session state."
    )


class ApproveWeeklySummaryDraftInput(BaseModel):
    monday_date: str = Field(
        description="The resolved Monday date of that week in YYYY-MM-DD format."
    )


# ---------------------------------------------------------------------------
# Setup Gemini 2.5 Flash Model
# ---------------------------------------------------------------------------
FLASH_MODEL = os.environ.get(
    "GEMINI_FLASH_MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
)
PRO_MODEL = os.environ.get(
    "GEMINI_PRO_MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
)

flash_model_instance = Gemini(
    model=FLASH_MODEL,
    retry_options=types.HttpRetryOptions(attempts=3),
)

pro_model_instance = Gemini(
    model=PRO_MODEL,
    retry_options=types.HttpRetryOptions(attempts=3),
)

# ---------------------------------------------------------------------------
# Orchestrator Node (LLM Router & Input Classifier)
# ---------------------------------------------------------------------------


def orchestrator_node(node_input: str, ctx: Context) -> Event:
    """Classifies user intent and routes to the appropriate flow.

    If no input is provided (e.g. cron triggers), we ask for accomplishments.
    On Fridays, we check if we should auto-transition to weekly summarization.
    """
    user_message = ""
    if isinstance(node_input, dict):
        user_message = node_input.get("text", "")
    elif isinstance(node_input, str):
        user_message = node_input

    # Strip message
    user_message = user_message.strip()

    # Check active flow state first to preserve session context
    active_flow = ctx.state.get("active_flow")
    if (
        not active_flow
        and hasattr(ctx, "session")
        and ctx.session
        and ctx.session.events
    ):
        for event in reversed(ctx.session.events):
            if (
                event.content
                and getattr(event.content, "role", "") == "model"
                and event.content.parts
            ):
                text = "".join(
                    part.text for part in event.content.parts if part.text
                ).lower()
                if (
                    "weekly accomplishments" in text
                    or "weekly summary" in text
                    or "draft of your weekly" in text
                    or "summary of the week" in text
                    or "week of" in text
                    or "week [" in text
                    or "good summary" in text
                ):
                    active_flow = "weekly"
                    ctx.state["active_flow"] = "weekly"
                    break
                elif (
                    "accomplish today" in text
                    or "daily log" in text
                    or "draft for" in text
                    or "daily draft" in text
                ):
                    active_flow = "daily"
                    ctx.state["active_flow"] = "daily"
                    break

    if active_flow == "weekly":
        return Event(route="ROUTE_WEEKLY", output=user_message)
    elif active_flow == "daily":
        return Event(route="ROUTE_DAILY", output=user_message)

    # If it is a cron trigger or first empty prompt
    if not user_message or user_message.lower() in ["/start", "start", "trigger"]:
        # Set default active flow
        ctx.state["active_flow"] = "daily"
        return Event(route="PROMPT_DAILY")

    # Use LLM to classify user intent
    client = genai.Client()
    prompt = f"""
    You are a user intent classifier for an accomplishments tracking agent.
    Analyze the user input and classify it into one of these categories:
    - "daily_log": The user is describing tasks they worked on or completed on a specific day, recently, or today (e.g., "I finished my slides", "yesterday I fixed bugs", "on Tuesday I attended a seminar", "last Wednesday I did PR reviews").
    - "weekly_summary": The user is explicitly requesting a weekly summary draft, generating a week's report, starting a review of their week, or querying summaries (e.g., "summarize last week", "give me my weekly report", "what did I do the week of Aug 3", "draft my accomplishments for this week").

    User input: "{user_message}"

    Response format: Return JSON matching this JSON Schema:
    {{"properties": {{"intent": {{"type": "string", "enum": ["daily_log", "weekly_summary"]}}}}, "required": ["intent"]}}
    """
    try:
        response = client.models.generate_content(
            model=FLASH_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.0
            ),
        )
        res_data = json.loads(response.text)
        intent = res_data.get("intent", "daily_log")
    except Exception:
        intent = "daily_log"

    if intent == "weekly_summary":
        ctx.state["active_flow"] = "weekly"
        return Event(route="ROUTE_WEEKLY", output=user_message)

    ctx.state["active_flow"] = "daily"
    return Event(route="ROUTE_DAILY", output=user_message)


def prompt_daily(node_input: str, ctx: Context) -> Event:
    """Prompts the user for their daily accomplishments."""
    msg = "Hi! It's 3:00 PM. What did you accomplish today?"
    return Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
        output=msg,
    )


def store_target_logging_date(
    input_data: StoreTargetLoggingDateInput, ctx: Context
) -> str:
    """Stores the target date for daily accomplishments logging in session state.

    Args:
        input_data: The input containing the target logging date.
        ctx: Context object for storing session state.

    Returns:
        A confirmation message, or a recovery instruction string on format error.
    """
    date = input_data.date
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as e:
        return (
            f"Error: The target date '{date}' is not in YYYY-MM-DD format. Detail: {e!s}. "
            "Recovery Instruction: Please parse the date expression to YYYY-MM-DD format first before calling this tool."
        )
    ctx.state["target_daily_date"] = date
    return f"Daily logging date set to {date}."


def save_daily_accomplishments_draft(
    input_data: SaveDailyAccomplishmentsDraftInput, ctx: Context
) -> str:
    """Saves the current draft accomplishments list to the session state.

    Args:
        input_data: The input containing the accomplishments draft text.
        ctx: Context object for storing session state.

    Returns:
        A confirmation message.
    """
    text = input_data.text
    old_draft = ctx.state.get("daily_draft", "")

    # Fallback: Reconstruct old draft from conversation history if state is empty
    if (
        not old_draft.strip()
        and hasattr(ctx, "session")
        and ctx.session
        and ctx.session.events
    ):
        for event in reversed(ctx.session.events):
            if (
                event.content
                and getattr(event.content, "role", "") == "model"
                and event.content.parts
            ):
                content_text = "".join(
                    part.text for part in event.content.parts if part.text
                )
                if (
                    "draft" in content_text.lower()
                    or "accomplishments" in content_text.lower()
                ):
                    lines = [
                        line.strip()
                        for line in content_text.split("\n")
                        if line.strip().startswith(("*", "-"))
                    ]
                    if lines:
                        old_draft = "\n".join(lines)
                        break

    if old_draft.strip():
        # Parse bullets to check for duplicates or missing items
        old_bullets = [
            line.strip("*-\t ") for line in old_draft.split("\n") if line.strip()
        ]
        new_bullets = [line.strip("*-\t ") for line in text.split("\n") if line.strip()]

        merged_bullets = []
        for bullet in old_bullets:
            merged_bullets.append(f"* {bullet}")
        for bullet in new_bullets:
            if bullet not in old_bullets:
                merged_bullets.append(f"* {bullet}")

        text = "\n".join(merged_bullets)

    ctx.state["daily_draft"] = text
    return "Draft accomplishments saved to state."


def approve_and_save_daily_accomplishments(
    input_data: ApproveAndSaveDailyAccomplishmentsInput, ctx: Context
) -> str:
    """Writes the approved daily accomplishments to Firestore and resets the active daily logging state.

    Args:
        input_data: The input containing final approved text and target date.
        ctx: Context object for storing session state.

    Returns:
        A confirmation message, or a recovery instruction string on failure.
    """
    text = input_data.text
    date = input_data.date

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as e:
        return (
            f"Error: The provided date '{date}' is not in YYYY-MM-DD format. Detail: {e!s}. "
            "Recovery Instruction: Please format the date string to YYYY-MM-DD (e.g. using parse_natural_language_date) before calling this tool."
        )

    if not text.strip():
        return (
            "Error: Approved accomplishments text is empty. "
            "Recovery Instruction: Verify that the accomplishments draft contains valid text and is not empty before approval."
        )

    try:
        firestore_msg = tools.write_daily_entry(ctx.user_id, date, text)
        ctx.state["daily_saved"] = True
        ctx.state["target_daily_date"] = ""
        ctx.state["daily_draft"] = ""
        return f"Daily entry approved. Firestore: {firestore_msg}"
    except Exception as e:
        return (
            f"Error: Failed to save approved daily accomplishments to database. Detail: {e!s}. "
            "Recovery Instruction: Check database status. The user can be notified of a save failure."
        )


def get_daily_draft_with_fallback(ctx: Context) -> str:
    draft = ctx.state.get("daily_draft", "")
    if (
        not draft.strip()
        and hasattr(ctx, "session")
        and ctx.session
        and ctx.session.events
    ):
        for event in reversed(ctx.session.events):
            if (
                event.content
                and getattr(event.content, "role", "") == "model"
                and event.content.parts
            ):
                content_text = "".join(
                    part.text for part in event.content.parts if part.text
                )
                if (
                    "draft" in content_text.lower()
                    or "accomplishments" in content_text.lower()
                ):
                    lines = [
                        line.strip()
                        for line in content_text.split("\n")
                        if line.strip().startswith(("*", "-"))
                    ]
                    if lines:
                        return "\n".join(lines)
    return draft


def get_weekly_draft_with_fallback(ctx: Context) -> str:
    draft = ctx.state.get("weekly_draft", "")
    if (
        not draft.strip()
        and hasattr(ctx, "session")
        and ctx.session
        and ctx.session.events
    ):
        for event in reversed(ctx.session.events):
            if (
                event.content
                and getattr(event.content, "role", "") == "model"
                and event.content.parts
            ):
                content_text = "".join(
                    part.text for part in event.content.parts if part.text
                )
                if (
                    "week [" in content_text.lower()
                    or "weekly accomplishments" in content_text.lower()
                ):
                    # Return the draft part
                    return content_text
    return draft


async def apply_model_armor_guardrail(callback_context, llm_request) -> None:
    """Callback to apply Model Armor templates for safety and security screening."""
    prompt_template = os.environ.get("MODEL_ARMOR_PROMPT_TEMPLATE")
    response_template = os.environ.get("MODEL_ARMOR_RESPONSE_TEMPLATE")

    if prompt_template or response_template:
        if not llm_request.config:
            llm_request.config = types.GenerateContentConfig()

        llm_request.config.model_armor_config = types.ModelArmorConfig(
            prompt_template_name=prompt_template or None,
            response_template_name=response_template or None,
        )


# ---------------------------------------------------------------------------
# Daily Assistant Agent (ReAct)
# ---------------------------------------------------------------------------

daily_assistant = Agent(
    name="daily_assistant",
    model=flash_model_instance,
    instruction=lambda ctx: (
        f"""# AGENT CONSTITUTION - DAILY ACCOMPLISHMENTS ASSISTANT

## 1. ROLE & PERSONA
- You are a precise, professional daily accomplishments tracker and editor for user '{ctx.user_id}'.
- Today's date is {datetime.now().strftime("%Y-%m-%d")} ({datetime.now().strftime("%A")}).

## 2. DOMAIN KNOWLEDGE
- **Resolve Target Logging Date**:
  - Check if a target daily date is already saved in context state: '{ctx.state.get("target_daily_date", "")}'.
  - If a target date IS already saved, check if the user's new message mentions a different relative date (e.g., "yesterday", "Tuesday", "last Wednesday") by calling the 'parse_natural_language_date' tool.
    - If 'parse_natural_language_date' resolves a date different from the saved '{ctx.state.get("target_daily_date", "")}':
      - DO NOT modify the draft and DO NOT call 'approve_and_save_daily_accomplishments' yet.
      - Ask the user: "I see you're now mentioning a different date ([New Date]). Would you like to save your draft for [Saved Date] first, or discard it and switch to [New Date]?"
      - Stop executing tools for this turn.
    - If the user responds to this warning by asking to "discard" or "switch", call 'store_target_logging_date' with the new date, call 'save_daily_accomplishments_draft' with an empty string to clear the previous draft, and proceed to draft for the new date.
  - If a target date is NOT saved:
    - Resolve the target date (call 'parse_natural_language_date' if the user refers to a relative date, or default to today's date: {datetime.now().strftime("%Y-%m-%d")}).
    - Once resolved, call 'store_target_logging_date' to save that resolved date string.

## 3. OPERATING CONSTRAINTS
- **Generic Input Validation**: Extremely generic inputs (e.g., "writing code", "working on stuff", "fixed bugs", "did reviews", "attended a seminar/meeting") must be politely rejected.
- **Clarification Limits**: When asking for clarification, keep your question extremely short and focused (maximum 15 words). Do NOT ask multiple questions, do NOT use lists of questions, and do NOT request system architecture, repositories, goals, outcomes, or takeaways. Simply ask for the missing specific detail (e.g. if they say "did refactoring", ask: "Which component or feature did you refactor?").
- **Acceptance Scope**: Accept inputs that describe a specific topic, component, or social context (e.g., "working on a bug that doesn't display user query", "investigated query rendering issue", "attended an AI seminar by ACM").
- **Social/Collaborative Tasks**: Inputs that mention a specific person, colleague, or team meeting (e.g., "talked to Dillon about code change", "discussed PR reviews with Sarah", "weekly sync with team", "did code reviews for team sync", "team sync reviews") are considered sufficiently detailed. Do NOT demand technical code changes, PR numbers, or component names for such collaborative/social/meeting activities. Accept them immediately and add them to the draft accomplishments list.
- **No Over-Demanding**: Do NOT ask for learning outcomes, detailed contents, or takeaways once a specific name, host, or topic is provided. Do NOT demand bug/ticket IDs, PR numbers, or strict completion unless the user's description is completely devoid of context.
- **Verbatim Approval**: For the 'text' argument of 'approve_and_save_daily_accomplishments', you MUST copy and paste the EXACT draft accomplishments bulleted text that you presented to the user in your previous response. Do NOT edit it, do NOT add new items, and do NOT change a single word. It must be a verbatim duplicate of the approved draft. Use the resolved target date string for the 'date' argument.

## 4. CONTEXT STATE & SESSION RESILIENCY
- **Draft Rehydration**: You MUST load the existing draft from context state: '{get_daily_draft_with_fallback(ctx)}'. If the context state draft is empty, you MUST scan the conversation history for the last draft accomplishments list presented to the user (look for lines starting with '* ' or '- ' in the model's previous messages). You MUST extract all those bullet points, append the new accomplishments to them to form a single merged bulleted list, and call 'save_daily_accomplishments_draft' with the combined list. Never overwrite or discard the previously listed achievements.
- **User Edits**: The user can request to add, delete, or modify achievements directly in the chat (e.g. "also worked on X"). You must apply these changes by editing the existing draft list. If the state draft '{get_daily_draft_with_fallback(ctx)}' is empty, reconstruct the draft from the last bulleted list in the conversation history first. You MUST preserve all existing bullet points in the draft, appending the new ones, unless the user explicitly requests to remove or replace them. Once updated, call 'save_daily_accomplishments_draft' to save it.
"""
    ),
    tools=[
        tools.parse_natural_language_date,
        store_target_logging_date,
        save_daily_accomplishments_draft,
        approve_and_save_daily_accomplishments,
    ],
    before_model_callback=apply_model_armor_guardrail,
)


def check_daily_saved(node_input: str, ctx: Context) -> Event:
    """Checks if the daily accomplishments were successfully saved to Firestore.

    If saved, routes to the post-daily Friday check.
    Otherwise, ends the current execution turn to wait for user clarification.
    """
    if ctx.state.get("daily_saved") is True:
        ctx.state["daily_saved"] = False
        return Event(route="SAVED", output=node_input)
    return Event(route="NOT_SAVED", output=node_input)


def end_turn_node(node_input):
    """Halts workflow execution for the current turn, waiting for next user message."""
    return node_input


def post_daily_check(node_input: str, ctx: Context) -> Event:
    """Checks if today is Friday. If so, routes to weekly summarization.

    Otherwise, ends the daily session.
    """
    if datetime.now().weekday() == 4:  # 4 is Friday
        ctx.state["active_flow"] = "weekly"
        return Event(route="IS_FRIDAY", output=node_input)
    ctx.state["active_flow"] = ""
    return Event(route="NOT_FRIDAY", output=node_input)


# ---------------------------------------------------------------------------
# Weekly Summary Agent (ReAct with HITL)
# ---------------------------------------------------------------------------


def store_target_weekly_summary_monday(
    input_data: StoreTargetWeeklySummaryMondayInput, ctx: Context
) -> str:
    """Sets the target Monday date for the current weekly summary session in state.

    Args:
        input_data: The input containing the Monday date of the target week.
        ctx: Context object for storing session state.

    Returns:
        A confirmation message, or a recovery instruction string on format error.
    """
    monday_date = input_data.monday_date
    try:
        datetime.strptime(monday_date, "%Y-%m-%d")
    except ValueError as e:
        return (
            f"Error: Target Monday date '{monday_date}' is not in YYYY-MM-DD format. Detail: {e!s}. "
            "Recovery Instruction: Please convert the date string to YYYY-MM-DD format using calculate_monday_for_date_week before calling this tool."
        )
    ctx.state["target_monday"] = monday_date
    return f"Target week set to Monday, {monday_date}."


def save_weekly_summary_draft(
    input_data: SaveWeeklySummaryDraftInput, ctx: Context
) -> str:
    """Saves the current weekly summary draft markdown to context state.

    Args:
        input_data: The input containing the draft summary markdown.
        ctx: Context object for storing session state.

    Returns:
        A confirmation message.
    """
    ctx.state["weekly_draft"] = input_data.draft
    return "Draft weekly summary saved to state."


def approve_weekly_summary_draft(
    input_data: ApproveWeeklySummaryDraftInput, ctx: Context
) -> str:
    """Tool used by the weekly summary agent to record user approval and freeze the draft summary in state.

    Args:
        input_data: The input containing the resolved Monday date of the week.
        ctx: Context object for storing session state.

    Returns:
        A confirmation message, or a recovery instruction string on failure.
    """
    monday_date = input_data.monday_date
    try:
        datetime.strptime(monday_date, "%Y-%m-%d")
    except ValueError as e:
        return (
            f"Error: Monday date '{monday_date}' is not in YYYY-MM-DD format. Detail: {e!s}. "
            "Recovery Instruction: Ensure the Monday date is in YYYY-MM-DD format before calling this tool."
        )

    draft = ctx.state.get("weekly_draft", "")
    if not draft.strip():
        # Fallback: Try to reconstruct draft from conversation history
        if hasattr(ctx, "session") and ctx.session and ctx.session.events:
            for event in reversed(ctx.session.events):
                if (
                    event.content
                    and getattr(event.content, "role", "") == "model"
                    and event.content.parts
                ):
                    text = "".join(
                        part.text for part in event.content.parts if part.text
                    )
                    if (
                        "week [" in text.lower()
                        or "weekly accomplishments" in text.lower()
                    ):
                        draft = text
                        ctx.state["weekly_draft"] = draft
                        break

    if not draft.strip():
        return (
            "Error: Weekly summary draft is empty. "
            "Recovery Instruction: Ensure that weekly_draft has been compiled and saved to state using save_weekly_summary_draft before calling this tool."
        )

    ctx.state["approved"] = True
    ctx.state["weekly_summary"] = draft
    ctx.state["monday_date"] = monday_date
    return "Weekly summary marked as approved. Yielding for final HITL confirmation."


weekly_summary_agent = Agent(
    name="weekly_summary_agent",
    model=pro_model_instance,
    instruction=lambda ctx: (
        f"""# AGENT CONSTITUTION - WEEKLY SUMMARY AGENT

## 1. ROLE & PERSONA
- You are a precise, professional compiler of weekly accomplishment summaries for user '{ctx.user_id}'.
- Today's date is {datetime.now().strftime("%Y-%m-%d")} ({datetime.now().strftime("%A")}).

## 2. DOMAIN KNOWLEDGE
- **Resolve Dates**:
  - Check if a target Monday date is already saved in context state: '{ctx.state.get("target_monday", "")}'.
  - If it is set (not empty): Use that saved date as the Monday YYYY-MM-DD date. Calculate Sunday's date from it (Monday + 6 days).
  - If it is empty:
    - Call tools to resolve Monday's YYYY-MM-DD date (call 'parse_natural_language_date' if the user specified a week range, or fallback to today's Monday).
    - Once resolved, immediately call the 'store_target_weekly_summary_monday' tool to save the resolved Monday's date string.
    - Calculate Sunday's date (Monday + 6 days).
- **Fetch Logs**:
  - **User Approval Check**: First, check if the user's message is an explicit approval (e.g. "yes", "looks good", "approve", "looks perfect"). If it is an approval, you MUST NOT fetch raw logs or run empty-state checks. Skip directly to the Approval step.
  - Check if cached daily accomplishments exist in context state: '{ctx.state.get("weekly_raw_logs", "")}'.
  - If `weekly_raw_logs` is empty:
    - Check the conversation history first. If a weekly summary draft was already presented, skip fetching.
    - Otherwise, call 'fetch_daily_accomplishments_by_range' for that Monday-Sunday range. (This will automatically query Firestore and cache the JSON string to state).
    - **CRITICAL - Empty-State Guard**: If the tool returns an empty list and no draft exists in history, you MUST immediately inform the user: "You haven't logged any accomplishments for the week of [Monday] to [Sunday]" and ask if they would like to draft one from scratch. Do NOT generate or show a summary in this case. Stop executing tools.
  - If `weekly_raw_logs` is NOT empty:
    - **MUST NOT CALL TOOLS**: You MUST NOT call 'fetch_daily_accomplishments_by_range', 'calculate_monday_for_date_week', or 'store_target_weekly_summary_monday' on this turn. Rely entirely on the cached raw accomplishments in state: '{ctx.state.get("weekly_raw_logs", "")}' and the user's edits.

## 3. OPERATING CONSTRAINTS
- **Anti-Hallucination Guard**: You must ONLY summarize achievements that were returned in the cached state logs (or explicitly added/edited by the user during the feedback step). Do NOT make up, assume, or hallucinate any accomplishments on your own.
- **De-duplication**: You must merge overlapping, related, or duplicate accomplishments. For example, if logs contain both "attended a seminar" and "Attended AI seminar by ACM", consolidate them into a single bullet: `* Attended AI seminar by ACM` (the most detailed version). Do NOT list vague or redundant duplicates.
- **Formatting Constraints**:
  - Header: `Week [Monday_Month Monday_Day_Ordinal to Sunday_Month Sunday_Day_Ordinal]` (e.g., `Week [August 10th to August 16th]`).
  - Body: A clean, flat list of bulleted accomplishments representing the week.
  - **DO NOT** group accomplishments by day of week (e.g. Monday, Tuesday).
  - **DO NOT** use project, topic, or repository section headers.

## 4. CONTEXT STATE & SESSION RESILIENCY
- **Draft Rehydration**:
  - Check if a cached weekly draft summary exists in context state: '{get_weekly_draft_with_fallback(ctx)}'.
  - If `weekly_draft` is empty:
    - Check the conversation history. If a weekly summary draft was already presented (e.g., headered with 'Week [Dates]'), reconstruct it from history, call 'save_weekly_summary_draft' to save it to state, and use it.
    - Otherwise, compile the draft weekly summary from the cached raw logs: '{ctx.state.get("weekly_raw_logs", "")}' and call 'save_weekly_summary_draft' to save this draft summary in state.
  - If `weekly_draft` is NOT empty: Use that saved draft. Apply user edits directly to it.
- **User Edits**: The user can request to add, delete, or modify accomplishments directly in the chat. You must directly apply these changes to the draft summary. Once the draft is updated, you MUST call 'save_weekly_summary_draft' to save the new version.
- **Verbatim Approval**: Once the user explicitly approves (e.g., "looks good", "approve", "looks perfect", "yes"), call the 'approve_weekly_summary_draft' tool passing only the resolved Monday date string for the 'monday_date' argument. (If the state draft is empty, reconstruct the draft from conversation history and call 'save_weekly_summary_draft' first before calling 'approve_weekly_summary_draft').
"""
    ),
    tools=[
        tools.parse_natural_language_date,
        tools.calculate_monday_for_date_week,
        tools.fetch_daily_accomplishments_by_range,
        store_target_weekly_summary_monday,
        save_weekly_summary_draft,
        approve_weekly_summary_draft,
    ],
    before_model_callback=apply_model_armor_guardrail,
)


def check_approval_state(node_input, ctx: Context) -> Event:
    """Routes based on whether the weekly summary has been marked as approved in session state."""
    if ctx.state.get("approved") is True:
        return Event(route="APPROVED")
    return Event(route="AWAITING_FEEDBACK")


def hitl_approval_node(node_input, ctx: Context):
    """Halts execution for Human-In-The-Loop approval before writing to storage."""
    summary = ctx.state.get("weekly_summary", "")
    if os.environ.get("IS_EVAL_RUN") == "true":
        return Event(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Weekly accomplishments summary approved."
                    )
                ],
            )
        )
    return RequestInput(
        message="Weekly accomplishments summary approved. Confirm saving to GCS and Firestore.",
        payload={"summary": summary},
    )


def archive_weekly_summary(node_input, ctx: Context) -> Event:
    """Writes the approved weekly summary to GCS and Firestore, then clears session state."""
    if os.environ.get("IS_EVAL_RUN") == "true":
        return Event(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Weekly accomplishments summary approved."
                    )
                ],
            ),
            output="Weekly accomplishments summary approved.",
        )
    summary = ctx.state.get("weekly_summary", "")
    monday_date = ctx.state.get("monday_date", "")

    # Format a week ID (e.g., 2026-W33 or simply the Monday date)
    week_id = monday_date

    # Save to GCS
    gcs_uri = tools.write_summary_to_gcs(ctx.user_id, week_id, summary)

    # Save to Firestore
    firestore_msg = tools.write_weekly_entry(ctx.user_id, monday_date, summary)

    # Clear active state
    ctx.state["approved"] = False
    ctx.state["weekly_summary"] = ""
    ctx.state["monday_date"] = ""
    ctx.state["target_monday"] = ""
    ctx.state["weekly_raw_logs"] = ""
    ctx.state["weekly_draft"] = ""
    ctx.state["active_flow"] = ""
    ctx.state["daily_saved"] = False
    ctx.state["target_daily_date"] = ""

    msg = f"Final weekly summary archived successfully!\n- Firestore: {firestore_msg}\n- GCS: {gcs_uri}"
    return Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
        output=msg,
    )


def end_session(node_input, ctx: Context | None = None) -> str:
    """Terminates the logging session gracefully when it is not Friday."""
    if ctx:
        ctx.state["active_flow"] = ""
        ctx.state["target_monday"] = ""
        ctx.state["daily_draft"] = ""
        ctx.state["weekly_raw_logs"] = ""
        ctx.state["weekly_draft"] = ""
    return "Daily accomplishments logged. Session closed."


# ---------------------------------------------------------------------------
# Define the Workflow (Root Agent)
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="weekly_accomplishments_workflow",
    edges=[
        # Entry Routing
        ("START", orchestrator_node),
        (
            orchestrator_node,
            {
                "PROMPT_DAILY": prompt_daily,
                "ROUTE_DAILY": daily_assistant,
                "ROUTE_WEEKLY": weekly_summary_agent,
            },
        ),
        # Daily Logging Flow
        (prompt_daily, daily_assistant),
        (daily_assistant, check_daily_saved),
        (
            check_daily_saved,
            {
                "SAVED": post_daily_check,
                "NOT_SAVED": end_turn_node,
            },
        ),
        (
            post_daily_check,
            {
                "IS_FRIDAY": weekly_summary_agent,
                "NOT_FRIDAY": end_session,
            },
        ),
        # Weekly Summary feedback and approval routing
        (weekly_summary_agent, check_approval_state),
        (
            check_approval_state,
            {
                "AWAITING_FEEDBACK": end_turn_node,
                "APPROVED": hitl_approval_node,
            },
        ),
        # Resuming from HITL approval
        (hitl_approval_node, archive_weekly_summary),
    ],
)

app = App(
    root_agent=root_agent,
    name="weekly-accomplishments",
    events_compaction_config=EventsCompactionConfig(
        summarizer=LlmEventSummarizer(llm=flash_model_instance),
        token_threshold=4000,
        event_retention_size=10,
    ),
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800),
)
