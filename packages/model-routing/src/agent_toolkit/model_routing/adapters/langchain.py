"""LangChain Runnables behind a `ProviderCooldown`, so a fallback chain skips cooling providers.

Install with the `langchain` extra. `with_cooldown_fallbacks([(key, model), ...], cooldown)`
wraps each entry in a `CooldownRunnable` and tries them in order, like `with_fallbacks`: an entry
the breaker refuses is skipped without calling its provider, and a handled error moves on to the
next entry. Unlike `with_fallbacks`, a skip is never the error the caller gets while a provider
actually failed: when every entry fails or is refused, the first real error is raised, and
`ProviderCoolingDown` (of the first entry) only when every entry was refused. An error that is
not in `exceptions_to_handle` (by default every `Exception` is handled) is raised at once, so
pass a narrower tuple to stop on, say, a budget stop instead of trying the next provider.

Bind tools (`bind_tools`) or structured output on each model before wrapping it: the wrapper is a
plain Runnable, not a chat model. Its input and output schemas are the wrapped runnable's (the
first entry's, for the chain) and its config specs those of every entry. Each attempt runs in its
own config context, as in `with_fallbacks`, so tracing nests under the chain's run. `invoke`
and `ainvoke` go through the breaker; `batch`, `stream` and their async forms use LangChain's
defaults, which call `invoke`/`ainvoke` once per input (a stream yields the whole result as one
chunk).

Prefer `with_cooldown_fallbacks` to `first.with_fallbacks(rest)` over `CooldownRunnable`s: that
also skips cooling entries, but LangChain then raises the first error, which may be a skip that
hides the fallback's real failure.

Build the tracker with `classify=langchain_is_retryable` so a LangChain `ModelError` decides by
its own `is_retryable`: the core classifier does not know LangChain's exception types.
"""

from collections.abc import Sequence
from functools import partial
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.exceptions import ModelError
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import (
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
    set_config_context,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    coro_with_context,
    get_unique_config_specs,
)

from agent_toolkit.model_routing.cooldown import ProviderCooldown, ProviderCoolingDown, is_retryable

__all__ = [
    "CooldownFallbacks",
    "CooldownRunnable",
    "ProviderCoolingDown",
    "langchain_is_retryable",
    "with_cooldown_fallbacks",
]


def _decide_model_error(link: BaseException) -> bool | None:
    return link.is_retryable if isinstance(link, ModelError) else None


def langchain_is_retryable(
    error: BaseException,
    *,
    retryable: tuple[type[BaseException], ...] = (),
    never: tuple[type[BaseException], ...] = (),
) -> bool:
    """`is_retryable` where the first LangChain `ModelError` in the chain decides by its own
    `is_retryable` (a context overflow raised from a 500 is not retryable). Pass extra types with
    `functools.partial`, for example `never=(BudgetExceededError,)`."""
    return is_retryable(error, retryable=retryable, never=never, decide=_decide_model_error)


class _Delegating[Input, Output](Runnable[Input, Output]):
    """Reports the input and output types (so the schemas) and config specs of `_inner`."""

    _inner: Runnable[Input, Output]

    @property
    def InputType(self) -> type[Input]:
        return self._inner.InputType

    @property
    def OutputType(self) -> type[Output]:
        return self._inner.OutputType

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self._inner.config_specs


class CooldownRunnable[Input, Output](_Delegating[Input, Output]):
    """`runnable` behind `key`'s breaker: refused with `ProviderCoolingDown` while `cooldown`
    keeps `key` out, and every call's outcome is reported to it. `chain` is every key of the
    fallback chain, in order."""

    def __init__(
        self,
        runnable: Runnable[Input, Output],
        key: str,
        chain: Sequence[str],
        cooldown: ProviderCooldown,
    ) -> None:
        self._inner = runnable
        self._key = key
        self._chain = tuple(chain)
        self._cooldown = cooldown

    @property
    def key(self) -> str:
        return self._key

    def invoke(self, input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output:
        call = partial(self._inner.invoke, input, config, **kwargs)
        return self._cooldown.call(self._key, self._chain, call)

    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        call = partial(self._inner.ainvoke, input, config, **kwargs)
        return await self._cooldown.acall(self._key, self._chain, call)


class CooldownFallbacks[Input, Output](_Delegating[Input, Output]):
    """Tries `entries` in order; see the module docstring for which error it raises."""

    def __init__(
        self,
        entries: Sequence[CooldownRunnable[Input, Output]],
        *,
        exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        if not entries:
            raise ValueError("CooldownFallbacks needs at least one entry")
        self._entries = tuple(entries)
        self._inner = self._entries[0]
        self._handled: tuple[type[BaseException], ...] = (
            ProviderCoolingDown,
            *exceptions_to_handle,
        )

    @property
    def entries(self) -> tuple[CooldownRunnable[Input, Output], ...]:
        return self._entries

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec for entry in self._entries for spec in entry.config_specs
        )

    def invoke(self, input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output:
        config = ensure_config(config)
        run = get_callback_manager_for_config(config).on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        errors = _Errors()
        for entry in self._entries:
            try:
                child = _child(config, run)
                with set_config_context(child) as context:
                    output = context.run(entry.invoke, input, child, **kwargs)
            except self._handled as error:
                errors.add(error)
            except BaseException as error:
                run.on_chain_error(error)
                raise
            else:
                run.on_chain_end(output)
                return output
        failure = errors.to_raise()
        run.on_chain_error(failure)
        raise failure

    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        config = ensure_config(config)
        run = await get_async_callback_manager_for_config(config).on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        errors = _Errors()
        for entry in self._entries:
            try:
                child = _child(config, run)
                with set_config_context(child) as context:
                    coro = context.run(entry.ainvoke, input, child, **kwargs)
                    output = await coro_with_context(coro, context)
            except self._handled as error:
                errors.add(error)
            except BaseException as error:
                await run.on_chain_error(error)
                raise
            else:
                await run.on_chain_end(output)
                return output
        failure = errors.to_raise()
        await run.on_chain_error(failure)
        raise failure


class _Errors:
    """The first real error, else the first skip."""

    def __init__(self) -> None:
        self._failure: BaseException | None = None
        self._skip: ProviderCoolingDown | None = None

    def add(self, error: BaseException) -> None:
        if isinstance(error, ProviderCoolingDown):
            if self._skip is None:
                self._skip = error
        elif self._failure is None:
            self._failure = error

    def to_raise(self) -> BaseException:
        failure = self._failure if self._failure is not None else self._skip
        if failure is None:  # Unreachable: every entry adds an error before this is called.
            raise RuntimeError("no entry ran")
        return failure


def _child(
    config: RunnableConfig, run: CallbackManagerForChainRun | AsyncCallbackManagerForChainRun
) -> RunnableConfig:
    return patch_config(config, callbacks=run.get_child())


def with_cooldown_fallbacks[Input, Output](
    entries: Sequence[tuple[str, Runnable[Input, Output]]],
    cooldown: ProviderCooldown,
    *,
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
) -> CooldownFallbacks[Input, Output]:
    """`entries` (`(key, runnable)` pairs, in order) as a fallback chain, each behind `cooldown`.
    Keys must be unique and there must be at least one entry."""
    keys = tuple(key for key, _ in entries)
    if not keys:
        raise ValueError("with_cooldown_fallbacks needs at least one entry")
    if len(set(keys)) != len(keys):
        raise ValueError("with_cooldown_fallbacks keys must be unique")
    wrapped = [CooldownRunnable(runnable, key, keys, cooldown) for key, runnable in entries]
    return CooldownFallbacks(wrapped, exceptions_to_handle=exceptions_to_handle)
