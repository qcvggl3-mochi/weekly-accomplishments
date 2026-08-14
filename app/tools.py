import os
import re
from datetime import datetime, timedelta

from google import genai
from google.adk.agents.context import Context
from google.cloud import firestore, storage
from google.genai import types
from pydantic import BaseModel, Field


# Pydantic input and output schemas to validate tool arguments
class ParseDateInput(BaseModel):
    date_string: str = Field(
        description="The raw natural language date expression to parse (e.g., 'yesterday', 'last Tuesday', 'Tuesday')."
    )


class GetMondayInput(BaseModel):
    date_str: str = Field(
        description="The date string in YYYY-MM-DD format from which to calculate Monday's date."
    )


class FetchAccomplishmentsInput(BaseModel):
    start_date: str = Field(
        description="The start date in YYYY-MM-DD format (inclusive)."
    )
    end_date: str = Field(description="The end date in YYYY-MM-DD format (inclusive).")


class ParsedDate(BaseModel):
    date: str  # YYYY-MM-DD format


def _parse_relative_date_pure_python(date_string: str) -> str | None:
    # Normalize string
    s = date_string.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)  # Remove punctuation except hyphen

    # Check if already in YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    today = datetime.now()

    if s == "today":
        return today.strftime("%Y-%m-%d")
    if s in ["yesterday", "yest"]:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    # Check if weekday name is mentioned
    target_weekday = None
    is_last = "last" in s or "past" in s

    for day_name, day_idx in weekdays.items():
        if day_name in s:
            target_weekday = day_idx
            break

    if target_weekday is not None:
        today_weekday = today.weekday()
        # Calculate delta to most recent past occurrence of target_weekday
        delta = (today_weekday - target_weekday) % 7
        if delta == 0 and is_last:
            delta = 7
        elif delta == 0:
            delta = 0  # means today

        target_date = today - timedelta(days=delta)
        if is_last and delta != 7:
            # subtract an additional 7 days if "last Tuesday" was requested
            target_date = target_date - timedelta(days=7)

        return target_date.strftime("%Y-%m-%d")

    return None


def parse_natural_language_date(input_data: ParseDateInput) -> str:
    """Parses a natural language date expression (e.g. 'yesterday', 'last Tuesday') into YYYY-MM-DD format.

    Args:
        input_data: The parsed input containing the natural language date expression.

    Returns:
        The parsed date in YYYY-MM-DD format, or a recovery instruction string in case of error.
    """
    date_string = input_data.date_string
    if not date_string:
        return (
            "Error: The date_string argument is empty. "
            "Recovery Instruction: Please provide a non-empty relative or absolute date string (e.g., 'today', 'yesterday')."
        )
    try:
        # Fast path: Try parsing locally first
        if local_date := _parse_relative_date_pure_python(date_string):
            return local_date

        client = genai.Client()
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_day = datetime.now().strftime("%A")

        prompt = f"""
        You are a date extraction assistant for an accomplishments logging system.
        Given the user input: "{date_string}"
        And the reference date today is: {current_date} ({current_day})

        Parse the target date in YYYY-MM-DD format.

        CRITICAL RULE: Since this is for logging past achievements, all date references must resolve to PAST dates relative to today. When resolving weekday names (e.g., "Tuesday"), always resolve to the most recent past occurrence of that weekday (e.g., if today is Thursday Aug 13, "Tuesday" is Aug 11, NOT Aug 18).
        """

        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ParsedDate,
                temperature=0.0,
            ),
        )

        parsed = ParsedDate.model_validate_json(response.text)
        return parsed.date
    except Exception as e:
        return (
            f"Error: Failed to parse natural language date '{date_string}'. Detail: {e!s}. "
            "Recovery Instruction: Ask the user to clarify the target date in a simpler format, or default to today's date."
        )


def calculate_monday_for_date_week(input_data: GetMondayInput) -> str:
    """Calculates the YYYY-MM-DD date of the Monday of the week containing the input date.

    Args:
        input_data: The input containing the YYYY-MM-DD date string.

    Returns:
        The YYYY-MM-DD date string of the Monday of that week, or a recovery instruction string on error.
    """
    date_str = input_data.date_str
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        monday = dt - timedelta(days=dt.weekday())
        return monday.strftime("%Y-%m-%d")
    except ValueError as e:
        return (
            f"Error: The provided date '{date_str}' is not in YYYY-MM-DD format. Detail: {e!s}. "
            "Recovery Instruction: Please ensure you parse the date to YYYY-MM-DD format before calling this tool, "
            "or request a valid YYYY-MM-DD date string from the user."
        )
    except Exception as e:
        return (
            f"Error: An unexpected error occurred while calculating Monday for date '{date_str}'. Detail: {e!s}. "
            "Recovery Instruction: Verify the input date is correct and try again."
        )


def _get_firestore_collection() -> str:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        db = firestore.Client()
        project_id = db.project
    return f"{project_id}-firestore-collection"


def write_daily_entry(
    user_id: str, date: str, text: str, label: str = "daily", ctx: Context | None = None
) -> str:
    """Writes a daily accomplishment entry to the Firestore database.

    Args:
        user_id: The ID of the user.
        date: The date of the accomplishment in YYYY-MM-DD format.
        text: The accomplishment text.
        label: The label of the entry (defaults to 'daily').
        ctx: Optional ADK Context injection.

    Returns:
        A confirmation message.
    """
    db = firestore.Client()
    collection_name = _get_firestore_collection()
    doc_ref = (
        db.collection(collection_name)
        .document(user_id)
        .collection("accomplishments")
        .document(date)
    )
    doc_ref.set(
        {
            "date": date,
            "text": text,
            "label": label,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
    )
    if ctx:
        ctx.state["daily_saved"] = True
    return f"Successfully saved to accomplishments for {date}."


def fetch_daily_accomplishments_by_range(
    input_data: FetchAccomplishmentsInput, ctx: Context | None = None
) -> list[dict] | str:
    """Reads all daily accomplishment entries for the current user in a given date range (inclusive).

    Args:
        input_data: The input containing start and end dates.
        ctx: Optional ADK Context injection (automatically injected by the framework).

    Returns:
        A list of accomplishment dictionaries, or a recovery instruction string in case of errors.
    """
    if not ctx or not ctx.user_id:
        return (
            "Error: Active user session or context is missing. "
            "Recovery Instruction: Ensure you are running this within an active session with a valid user ID."
        )

    start_date = input_data.start_date
    end_date = input_data.end_date

    try:
        # Verify YYYY-MM-DD format
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return (
            f"Error: Start date '{start_date}' or end date '{end_date}' is not in YYYY-MM-DD format. Detail: {e!s}. "
            "Recovery Instruction: Ensure dates are parsed using 'parse_natural_language_date' prior to calling this tool."
        )

    if start_date > end_date:
        return (
            f"Error: Start date '{start_date}' is greater than end date '{end_date}'. "
            "Recovery Instruction: Swapping start_date and end_date is required or check dates calculations."
        )

    try:
        db = firestore.Client()
        collection_name = _get_firestore_collection()
        docs = (
            db.collection(collection_name)
            .document(ctx.user_id)
            .collection("accomplishments")
            .stream()
        )

        results = []
        for d in docs:
            data = d.to_dict()
            doc_date = data.get("date")
            if doc_date and start_date <= doc_date <= end_date:
                if data.get("label") == "daily":
                    results.append(
                        {
                            "date": doc_date,
                            "text": data.get("text"),
                            "label": data.get("label", "daily"),
                        }
                    )

        # Sort results chronologically by date
        results.sort(key=lambda x: x["date"])

        # Cache raw logs in session state
        import json

        ctx.state["weekly_raw_logs"] = json.dumps(results)

        return results
    except Exception as e:
        return (
            f"Error: Failed to fetch accomplishments from Firestore. Detail: {e!s}. "
            "Recovery Instruction: Check Firestore service status or user credentials. You can notify the user of a database connection error."
        )


def write_weekly_entry(
    user_id: str, date: str, text: str, label: str = "weekly"
) -> str:
    """Writes the weekly accomplishment summary to the Firestore database under the Monday date.

    Args:
        user_id: The ID of the user.
        date: The Monday date of the week in YYYY-MM-DD format.
        text: The final approved weekly summary markdown text.
        label: The label of the entry (defaults to 'weekly').

    Returns:
        A confirmation message.
    """
    db = firestore.Client()
    collection_name = _get_firestore_collection()
    doc_ref = (
        db.collection(collection_name)
        .document(user_id)
        .collection("accomplishments")
        .document(date)
    )
    doc_ref.set(
        {
            "date": date,
            "text": text,
            "label": label,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
    )
    return f"Successfully saved weekly summary to Firestore under {date}."


def write_summary_to_gcs(user_id: str, week_id: str, content: str) -> str:
    """Writes the finalized weekly summary markdown content to a GCS bucket.

    Args:
        user_id: The ID of the user.
        week_id: A string representing the week (e.g. '2026-W33' or Monday date).
        content: The weekly summary markdown content.

    Returns:
        The GCS URI of the written file.
    """
    client = storage.Client()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or client.project
    bucket_name = f"{project_id}-bucket"

    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        bucket = client.create_bucket(bucket_name)

    blob_path = f"users/{user_id}/weekly-summary-{week_id}.md"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(content, content_type="text/markdown")
    return f"gs://{bucket_name}/{blob_path}"
