"""E0-T02: importing `cli/app.py` exercises its module-level statements for coverage.

The CLI has no real commands yet (added by E6-T02/E6-T05/E8-T02); `main()` itself is
`# pragma: no cover` since exercising it would require actually invoking a Typer app with no
commands registered.
"""

import regimekit.cli.app


def test_e0_t02_cli_app_module_imports() -> None:
    """The module imports cleanly and exposes a Typer `app` and a `main` entry point."""
    assert callable(regimekit.cli.app.main)
    assert regimekit.cli.app.app is not None
