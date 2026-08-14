# Weekly Accomplishments Tracker Agent

An agentic AI assistant designed to help developers track daily achievements and automatically compile them into structured, de-duplicated weekly accomplishment reports.

This project is built using the Google **Agent Development Kit (ADK)** and the **A2A (Agent2Agent) Protocol**, allowing it to run as a backend service that seamlessly integrates with conversational interfaces or other collaborative agents.

---

## How It Works

The assistant operates via an orchestrator router that classifies user inputs and directs them to two core sub-agents:

### 1. Daily Logging Flow (`daily_assistant`)
* **Role:** A precise accomplishments logger.
* **Model:** Powered by `gemini-2.5-flash`.
* **Features:**
  * Resolves target logging dates by parsing relative language (e.g., *"yesterday"*, *"last Tuesday"*).
  * Validates daily inputs, politely rejecting vague tasks (e.g., *"fixed bugs"*, *"did coding"*) and asking short, focused clarification questions.
  * Persistently saves logs to **Firestore** under the user's document path.

### 2. Weekly Summarization Flow (`weekly_summary_agent`)
* **Role:** A compilation and formatting engine.
* **Model:** Powered by `gemini-2.5-pro`.
* **Features:**
  * Pulls raw daily entries from Firestore for the target Monday-Sunday date range.
  * Consolidates, de-duplicates, and merges related/overlapping achievements into a clean, flat list of accomplishments.
  * Supports interactive user feedback to edit the compiled weekly draft.
  * Yields for final Human-in-the-Loop (HITL) confirmation and approval before archiving.

---

## Project Structure

```
weekly-accomplishments/
├── app/         # Core agent code
│   ├── agent.py               # Main agent logic
│   ├── fast_api_app.py        # FastAPI Backend server
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, and load tests
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Antigravity CLI](https://antigravity.google/) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install `agents-cli` and its skills if not already installed:

```bash
uvx google-agents-cli setup
```

Install required packages:

```bash
agents-cli install
```

Test the agent with a local web server:

```bash
agents-cli playground
```

You can also use features from the [ADK](https://adk.dev/) CLI with `uv run adk`.

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        |
| `agents-cli deploy`  | Deploy agent to Cloud Run                                                                   || [A2A Inspector](https://github.com/a2aproject/a2a-inspector) | Launch A2A Protocol Inspector                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use the [A2A Inspector](https://github.com/a2aproject/a2a-inspector) to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.
