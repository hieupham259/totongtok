# Copilot custom instructions (repo-wide)

You are assisting with a **Python 3.10+** project that packages the `pyktok` library (source in `src/pyktok/pyktok.py`) plus small entrypoints (`main.py`, `app.py`).

## Goals
- Keep changes **minimal and surgical**; preserve existing behavior unless explicitly requested.
- Prefer **clarity and robustness** over cleverness.
- Maintain backwards compatibility for the public API in `src/pyktok/pyktok.py`.

## Project structure
- Library code: `src/pyktok/` (packaged as `pyktok` via `pyproject.toml`)
- CLI/script entrypoint examples: `main.py`
- Streamlit UI: `app.py` (runs via `streamlit run app.py`)
- Generated artifacts: `outputs/` (ignored by git; do not commit generated files)

## Packaging & dependencies
- Packaging: `hatchling` (see `pyproject.toml`)
- Runtime deps include `requests`, `beautifulsoup4`, `pandas`, `numpy`, `browser-cookie3`, `playwright`, `TikTokApi`, `streamlit`.
- Do **not** introduce new dependencies unless the user asks or it clearly reduces complexity.

## Coding conventions
- Use standard library modules where practical.
- Prefer explicit names over abbreviations.
- Favor pure functions/helpers when feasible, but avoid large refactors.
- Add/keep docstrings for user-facing functions.
- Use `os.path.join` / `pathlib` for paths (Windows-friendly). Never hardcode POSIX paths.

### API stability
When editing `src/pyktok/pyktok.py`:
- Avoid breaking function signatures.
- If you must extend behavior, prefer adding **optional keyword parameters with defaults**.
- Keep I/O conventions consistent with existing behavior (e.g., `outputs/videos/<entity>` and `outputs/images/<entity>` when `destination` is not provided).

## Network, scraping, and safety constraints
This project interacts with TikTok pages and unofficial/undocumented endpoints.
- Add timeouts to network calls when touching requests.
- Handle intermittent failures gracefully (return `None` / empty outputs consistent with existing patterns; avoid crashing callers).
- Avoid changes that bypass security controls or facilitate wrongdoing.
- Assume TikTok behavior can change; write defensive parsing code.

## Data/output handling
- Treat `outputs/` as **generated**. Do not write code that requires committed files there.
- Ensure directories are created before writing.
- Do not add large binary test fixtures.

## Testing & verification
- There may be limited or no automated tests. If adding tests is appropriate, prefer `pytest` style and keep them deterministic.
- For manual verification instructions, keep them short and use existing entrypoints:
  - `python main.py`
  - `streamlit run app.py`
  - `playwright install` (required for Playwright)

## Documentation
- If you change behavior, parameters, or file outputs, update `README.md` usage examples accordingly.
