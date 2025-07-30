# Plan

This was originally posted as a technical assessment to a well-known and respected AI company, Moonshot. The theme of the technical assessment is "agent architecture", and the position I'm being assessed for is AI Agent Infra 开发实习生 (Engineering Intern). Below was the original requirement, in Chinese:

> 希望你可以从中选择一 个，借助你所知最有效的方法进行调研与原型验证，撰写一份最能体现你专业水平的技术报告或 Demo ，将它发布在 GitHub 并将链接分享给我们。
> 使用 Python 搭建一个 Agent Workflow。
> a. 目前流行的 Agent Workflow 有哪些？
> b. Workflow 的 Structure 指什么？
> c. 如何让 Agent 最终输出符合预期的内容
> d. 在 Agent 执行过程中，应如何制定内容安全策略
> e. 如何进行 Tracing

Obviously, all requirements above must be met. The requirement mentions a report and a demo, and I'm doing both. Report is in `docs/meta/RESEARCH.md`, and we're working on the demo right now. But to impress the recruiters, I decided to go above and beyond the requirements. We're going to make a production-ready, test-proof, and well-documented Python SDK.

Basically, the assessment asks us to "rebuild LangChain". Although, I don't like LangChain for some reasons, including that its syntax is too verbose. Thus, we want to design the module to mimic Vercel AI SDK. The differences of these two SDKs are outlined in [](./ANALOGY.md).

## 1. Objectives & Scope

-   **Goal:** Ship a minimal, easy-to-use Python SDK that supports

    -   LLM chains & ReAct-style agents
    -   Tool registration/invocation
    -   Optional schema validation (via Pydantic)
    -   Safety hooks (moderation)
    -   Tracing & logging

-   **Out of scope (v1):** Built-in vector stores, multi-agent orchestration, GUI

## 2. Architecture & Core Components

1. **Core Engine (`ai_core`)**

    - `LLMClient` interface + OpenAI driver

    - Prompt templating module
    - Simple in-memory `Session` for state/memory

2. **Chains Module (`ai_chains`)**

    - `Chain` class: sequence of prompt→LLM calls

3. **Agent Module (`ai_agents`)**

    - `Agent` class: loop of `think→act(tool)→observe` until stop
    - Tool registry + invocation API

4. **Validation Hooks (`ai_validation`)**

    - Optional Pydantic schema enforcement on outputs

5. **Safety & Moderation (`ai_safety`)**

    - Pre/post callbacks
    - Built-in OpenAI Moderation wrapper

6. **Tracing & Logging (`ai_tracing`)**

    - Simple step recorder (prompt, response, tool, timestamp)
    - Adapter for OpenTelemetry (optional extra)

## 3. Milestones & Timeline

| Phase                           | Deliverables                                                     | Est. Duration |
| ------------------------------- | ---------------------------------------------------------------- | ------------- |
| **A. Setup & Scaffolding**      | Repo init, folder layout, CI pipelines, linting, testing         | 1–2 days      |
| **B. Core Engine & LLMClient**  | `LLMClient` interface, OpenAI driver, prompt templating, Session | 2–3 days      |
| **C. Chains Module**            | `Chain` class, simple examples/tests                             | 1–2 days      |
| **D. Agents Module**            | `Agent` class, tool registry, basic ReAct loop                   | 3–4 days      |
| **E. Validation Hooks**         | Pydantic integration, auto-retry logic                           | 1–2 days      |
| **F. Safety Hooks**             | Moderation API wrapper, pre/post callbacks                       | 1–2 days      |
| **G. Tracing & Logging**        | Step recorder, log format, OpenTelemetry hook                    | 1–2 days      |
| **H. Documentation & Examples** | README, Quickstart, examples/, playground notebook               | 2 days        |
| **I. Release & Publish**        | Build artifacts, PyPI release, tag, announce                     | 1 day         |

_Total: \~2–3 weeks_ (assuming part-time or single developer)

## 4. Detailed Task Breakdown

### A. Setup & Scaffolding

-   Choose **Cookiecutter-pypackage** or **PyScaffold** template
-   Enable: pytest, black, isort, mypy, GitHub Actions for test/lint
-   Create `src/ai_sdk/` and empty modules

### B. Core Engine

-   Define `LLMClient` abstract base
-   Implement `OpenAIClient` with `chat()` & `complete()`
-   Build `Prompt` helper for f-string templates + role merging
-   Create `Session` object for carrying memory

### C. Chains Module

-   `Chain` API:

    ```python
    chain = Chain().add_prompt("Hello {name}").run(name="Alice")
    ```

-   Support sync/async execution

### D. Agents Module

-   Design `@tool` decorator & registry
-   `Agent` API:

    ```python
    agent = Agent().with_tools(search, calc)
    result = agent.run("Find X then compute Y")
    ```

-   Implement loop: LLM “action JSON” → dispatch tool → feed output → repeat

### E. Validation Hooks

-   Accept optional `schema: BaseModel` on `.run()`
-   After each step or final answer, call `schema.parse_obj()`
-   On failure, retry prompt with “repair” instruction (up to N times)

### F. Safety Hooks

-   Define callback hooks lists: `pre_llm`, `post_llm`
-   Provide `ai_safety.moderation` callback using OpenAI Moderation API
-   Wire default `post_llm` to run moderation and raise on violation

### G. Tracing & Logging

-   Create `Trace` dataclass for each step (prompt, response, tool, time)
-   Store in `Session.trace`
-   Expose `get_last_trace()`
-   Optionally, emit OpenTelemetry spans in callbacks

### H. Documentation & Examples

-   Write Quickstart in README (cover chain, agent, tool, schema, safety, trace)
-   Add `examples/simple_agent.py`
-   Provide Jupyter notebook in `playground/` demonstrating end-to-end

### I. Release

-   Bump version, tag in Git
-   Build wheel/sdist with `poetry build` or `python -m build`
-   Publish to Test PyPI, then to main PyPI
-   Announce in repo README and optionally via blog

## 5. Risks & Mitigations

-   **Scope creep:** Keep v1 minimal—focus on core API and examples.
-   **API design churn:** Lock down public API early; use feature flags for internal changes.
-   **Dependency conflicts:** Use optional extras and pin core dependencies.
-   **Validation reliability:** Limit auto-retry loops to avoid infinite cycles.

This plan gives you a clear roadmap: start with solid scaffolding and CI, implement each module in order of dependency, and finish with docs and release. Adjust the pacing as needed based on team size and priorities.
