"""E0-T05 AC3: error hierarchy, `path=` message prefixing, and `EXIT_CODES`."""

import pytest

from regimekit.cli.app import EXIT_CODES, exit_code_of
from regimekit.finance.errors import ProviderError
from regimekit.utils.errors import (
    ConfigError,
    DataError,
    NumericalError,
    PersistenceError,
    RegimekitError,
)

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)


@pytest.mark.parametrize(
    "cls", [RegimekitError, ConfigError, DataError, ProviderError, NumericalError, PersistenceError]
)
def test_e0_t05_error_message_prefixed_with_path(cls: type[RegimekitError]) -> None:
    """`Error("bad", path="model.sigma")` message starts with `model.sigma:`."""
    error = cls("bad", path="model.sigma")
    assert str(error).startswith("model.sigma:")


def test_e0_t05_error_message_without_path_is_unprefixed() -> None:
    """With no `path`, the message is not prefixed."""
    error = NumericalError("bad")
    assert str(error) == "bad"


@pytest.mark.parametrize("cls", [ConfigError, DataError, ProviderError, NumericalError])
def test_e0_t05_error_subclasses_are_regimekit_errors(cls: type[RegimekitError]) -> None:
    """Every subclass is-a `RegimekitError`."""
    assert issubclass(cls, RegimekitError)


def test_e0_t05_exit_codes_complete() -> None:
    """The core codes: config 2, data 3, numerical 4, persistence 1 (`ProviderError` -> 5 is finance-only)."""
    core = {ConfigError: 2, DataError: 3, NumericalError: 4, PersistenceError: 1}
    assert core.items() <= EXIT_CODES.items()


def test_e0_t05_provider_error_is_a_data_error_with_exit_code_5_in_the_finance_cli() -> None:
    """`ProviderError` subclasses `DataError`; the finance CLI maps it to 5, the most specific class wins."""
    import regimekit.finance.cli  # noqa: F401  (registers `ProviderError -> 5`)

    assert issubclass(ProviderError, DataError)
    assert exit_code_of(ProviderError("x")) == 5
    assert exit_code_of(DataError("x")) == 3
