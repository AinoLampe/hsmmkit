"""E0-T05 AC3: error hierarchy, `path=` message prefixing, and `EXIT_CODES`."""

import pytest

from regimekit.utils.errors import (
    EXIT_CODES,
    ConfigError,
    DataError,
    NumericalError,
    PersistenceError,
    ProviderError,
    RegimekitError,
)


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
    """`EXIT_CODES` matches overview §4.4: config=2, data=3, numerical=4, provider=5."""
    assert {
        ConfigError: 2,
        DataError: 3,
        NumericalError: 4,
        ProviderError: 5,
    } == EXIT_CODES
