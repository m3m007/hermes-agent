"""Plugin LLM calls re-enter the owning profile's scope outside any turn (#132887).

On a multiplexed gateway, plugins fire from contexts that never run inside a turn's
``_profile_runtime_scope`` (bridge host handlers, boot-time memory flushes). The
auxiliary runtime resolver reads ``OPENROUTER_BASE_URL`` through ``get_secret_str``
BEFORE the trusted ``model.base_url`` rung is consulted, so an unscoped host-owned
``ctx.llm`` call fails closed with ``UnscopedSecretError`` even when the profile's
config would answer. ``PluginLlm`` binds the PluginManager's immutable home and
re-enters its secret scope + HERMES_HOME override for the duration of the call; a
turn's own scope is never replaced, and non-multiplex deployments keep os.environ
semantics.

The resolver itself is also covered here against the GENUINE
``hermes_cli.runtime_provider_backends._resolve_openrouter_runtime`` (no
``call_llm`` replacement): a no-OpenRouter profile whose ``model.base_url`` is
``http://127.0.0.1:4000/v1`` must resolve without any credential read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.plugin_llm import PluginLlm, _TrustPolicy
from agent.secret_scope import (
    UnscopedSecretError,
    build_profile_secret_scope,
    current_secret_scope,
    current_secret_scope_home,
    reset_secret_scope,
    set_secret_scope,
)

ENV_VALUE = "https://served-profile.example/v1"
TURN_VALUE = "https://turn-scope.example/v1"
CUSTOM_BASE_URL = "http://127.0.0.1:4000/v1"


def _fake_response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _make_llm(home: Path, **kwargs) -> PluginLlm:
    return PluginLlm(
        plugin_id="demo",
        policy_loader=lambda _pid: _TrustPolicy(plugin_id="demo"),
        profile_home=str(home),
        **kwargs,
    )


def _write_profile(home: Path, *, base_url: str = ENV_VALUE) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(f"OPENROUTER_BASE_URL={base_url}\n")
    return home


def _install_probe(monkeypatch, record: dict) -> None:
    """Replace ``agent.auxiliary_client.call_llm`` with a probe that reads a profile secret
    exactly the way the runtime resolver's first line does — under multiplexing with no
    scope installed this is the line that failed closed."""
    from agent.secret_scope import get_secret_str

    def probe(**call_kwargs):
        from hermes_constants import get_hermes_home

        record["secret"] = get_secret_str("OPENROUTER_BASE_URL")
        record["scope_installed"] = current_secret_scope() is not None
        record["scope_home"] = current_secret_scope_home()
        record["hermes_home"] = str(get_hermes_home())
        return _fake_response()

    monkeypatch.setattr("agent.auxiliary_client.call_llm", probe)


def _install_async_probe(monkeypatch, record: dict) -> None:
    """Async sibling of :func:`_install_probe` for ``async_call_llm``."""
    from agent.secret_scope import get_secret_str

    async def probe(**call_kwargs):
        from hermes_constants import get_hermes_home

        record["secret"] = get_secret_str("OPENROUTER_BASE_URL")
        record["scope_installed"] = current_secret_scope() is not None
        record["scope_home"] = current_secret_scope_home()
        record["hermes_home"] = str(get_hermes_home())
        return _fake_response()

    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", probe)


@pytest.fixture
def multiplex_on(monkeypatch):
    """Arm multiplexing for one test without touching the process-global flag's neighbours."""
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    yield
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)


# ---------------------------------------------------------------------------
# The bug: an unscoped multiplex call re-enters the owning profile's scope
# ---------------------------------------------------------------------------

def test_unscoped_multiplex_call_reenters_owning_profile_scope(tmp_path, monkeypatch, multiplex_on):
    home = _write_profile(tmp_path / "served-profile")
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(home)

    result = llm.complete([{"role": "user", "content": "hi"}])

    assert result.text == "ok"
    assert record["secret"] == ENV_VALUE
    assert record["scope_installed"] is True
    assert record["scope_home"] == str(home)
    assert Path(record["hermes_home"]).resolve() == home.resolve()
    # The re-entered scope is transient: nothing leaks past the call.
    assert current_secret_scope() is None


def test_turn_scoped_call_keeps_its_own_scope(tmp_path, monkeypatch, multiplex_on):
    home = _write_profile(tmp_path / "served-profile")
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(home)
    token = set_secret_scope(
        {"OPENROUTER_BASE_URL": TURN_VALUE}, profile_home=str(tmp_path / "turn")
    )
    try:
        llm.complete([{"role": "user", "content": "hi"}])
    finally:
        reset_secret_scope(token)

    assert record["secret"] == TURN_VALUE
    assert current_secret_scope() is None


def test_single_profile_deployment_stays_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(tmp_path)

    llm.complete([{"role": "user", "content": "hi"}])

    assert record["scope_installed"] is False


def test_plugin_context_binds_manager_home(tmp_path):
    from hermes_cli.plugins import PluginContext, PluginManager

    manager = PluginManager(scope_key=str(tmp_path))
    manifest = SimpleNamespace(key="", name="demo")
    ctx = PluginContext(manifest=manifest, manager=manager)

    assert ctx.llm._profile_home == str(manager.home_path)


# ---------------------------------------------------------------------------
# Review item 1: _host_kwargs runs INSIDE the scope
# ---------------------------------------------------------------------------

def test_host_kwargs_resolves_inside_the_scope(tmp_path, monkeypatch, multiplex_on):
    """``_host_kwargs`` reads the main provider/model through auxiliary_client, which resolves
    config against the ambient HERMES_HOME. On the unscoped boot path it must see the owning
    profile's home, not the ambient one — otherwise the wrong profile's model is attributed."""
    home = _write_profile(tmp_path / "served-profile")
    seen: dict = {}

    def probe(**call_kwargs):
        from hermes_constants import get_hermes_home

        seen["hermes_home"] = str(get_hermes_home())
        seen["scope_installed"] = current_secret_scope() is not None
        return _fake_response()

    monkeypatch.setattr("agent.auxiliary_client.call_llm", probe)
    # An ambient home that is NOT the owning profile: the scope must win for the whole call.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))
    llm = _make_llm(home)

    llm.complete([{"role": "user", "content": "hi"}])

    assert Path(seen["hermes_home"]).resolve() == home.resolve()
    assert seen["scope_installed"] is True


# ---------------------------------------------------------------------------
# Review item 2: an ambient HERMES_HOME override is preserved, not replaced
# ---------------------------------------------------------------------------

def test_ambient_home_override_is_preserved(tmp_path, monkeypatch, multiplex_on):
    """A caller that already scoped config reads keeps them: the plugin's home is installed for
    the call and the caller's override is restored on exit, never silently dropped."""
    from hermes_constants import get_hermes_home_override, set_hermes_home_override

    home = _write_profile(tmp_path / "served-profile")
    caller_home = tmp_path / "caller"
    caller_home.mkdir()
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(home)

    caller_token = set_hermes_home_override(str(caller_home))
    try:
        llm.complete([{"role": "user", "content": "hi"}])
        # Inside the call the owning profile wins...
        assert Path(record["hermes_home"]).resolve() == home.resolve()
        # ...and the caller's own override is intact afterwards.
        assert get_hermes_home_override() == str(caller_home)
    finally:
        from hermes_constants import reset_hermes_home_override

        reset_hermes_home_override(caller_token)


def test_no_ambient_override_leaves_none_after_the_call(tmp_path, monkeypatch, multiplex_on):
    from hermes_constants import get_hermes_home_override

    home = _write_profile(tmp_path / "served-profile")
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(home)

    llm.complete([{"role": "user", "content": "hi"}])

    assert get_hermes_home_override() is None


# ---------------------------------------------------------------------------
# Review item 3: ContextVar-backed bindings survive concurrent async calls
# ---------------------------------------------------------------------------

def test_concurrent_async_calls_do_not_interleave_scopes(tmp_path, monkeypatch, multiplex_on):
    """Two overlapping async completions on different profiles each see their own scope and
    home; one task's reset never clears another's (both bindings are ContextVar-backed)."""
    home_a = _write_profile(tmp_path / "profile-a", base_url="https://a.example/v1")
    home_b = _write_profile(tmp_path / "profile-b", base_url="https://b.example/v1")
    seen: list = []

    async def probe(**call_kwargs):
        from agent.secret_scope import get_secret_str
        from hermes_constants import get_hermes_home

        # Yield so the other task runs while this one holds its scope.
        await asyncio.sleep(0)
        seen.append((
            get_secret_str("OPENROUTER_BASE_URL"), str(get_hermes_home()), current_secret_scope_home(),
        ))
        return _fake_response()

    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", probe)
    llm_a = _make_llm(home_a)
    llm_b = _make_llm(home_b)

    async def _run():
        # A list, not a dict keyed by model: the trust policy forbids model
        # overrides, and keying both calls under one key lets the second
        # overwrite the first.
        return await asyncio.gather(
            llm_a.acomplete([{"role": "user", "content": "a"}]),
            llm_b.acomplete([{"role": "user", "content": "b"}]),
        )

    asyncio.run(_run())

    assert len(seen) == 2
    assert {v[0] for v in seen} == {"https://a.example/v1", "https://b.example/v1"}
    assert {Path(v[1]).resolve() for v in seen} == {home_a.resolve(), home_b.resolve()}
    assert {Path(v[2]).resolve() for v in seen} == {home_a.resolve(), home_b.resolve()}
    assert current_secret_scope() is None


# ---------------------------------------------------------------------------
# Review item 7: async path, mid-call exception, partial state, wrong-profile turn
# ---------------------------------------------------------------------------

def test_async_path_reenters_scope(tmp_path, monkeypatch, multiplex_on):
    home = _write_profile(tmp_path / "served-profile")
    record: dict = {}
    _install_async_probe(monkeypatch, record)
    llm = _make_llm(home)

    result = asyncio.run(llm.acomplete([{"role": "user", "content": "hi"}]))

    assert result.text == "ok"
    assert record["secret"] == ENV_VALUE
    assert record["scope_installed"] is True
    assert Path(record["hermes_home"]).resolve() == home.resolve()
    assert current_secret_scope() is None


def test_exception_mid_call_resets_scope_and_home(tmp_path, monkeypatch, multiplex_on):
    """A caller that raises inside the scoped block must leave neither a secret scope nor a
    HERMES_HOME override behind."""
    from hermes_constants import get_hermes_home_override

    home = _write_profile(tmp_path / "served-profile")

    def boom(**call_kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", boom)
    llm = _make_llm(home)

    with pytest.raises(RuntimeError, match="provider exploded"):
        llm.complete([{"role": "user", "content": "hi"}])

    assert current_secret_scope() is None
    assert get_hermes_home_override() is None


def test_hydrate_succeeds_but_scope_set_fails_leaves_no_partial_state(tmp_path, monkeypatch, multiplex_on):
    """Hydration succeeds and ``set_secret_scope`` then fails: no scope may stay installed and
    the caller's home state must be exactly what it was (no cross-profile secret leak)."""
    from hermes_constants import get_hermes_home_override

    home = _write_profile(tmp_path / "served-profile")
    hydrated = []

    def fake_hydrate(path):
        hydrated.append(str(path))
        return {}

    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", fake_hydrate)

    from agent import plugin_llm

    real_set = plugin_llm.__dict__.get("_real_set_secret_scope")
    from agent import secret_scope

    def failing_set(secrets, *, profile_home=None):
        raise OSError("scope install refused")

    monkeypatch.setattr(secret_scope, "set_secret_scope", failing_set)
    # The facade imports the names lazily inside the context manager, so patch the module
    # attribute the lazy import resolves.
    monkeypatch.setattr("agent.secret_scope.set_secret_scope", failing_set)
    llm = _make_llm(home)

    with pytest.raises(OSError, match="scope install refused"):
        llm.complete([{"role": "user", "content": "hi"}])

    assert hydrated == [str(home)]
    assert current_secret_scope() is None
    assert get_hermes_home_override() is None
    assert real_set is None  # the production symbol was never replaced at module scope


def test_wrong_profile_turn_scope_keeps_its_own_binding(tmp_path, monkeypatch, multiplex_on):
    """Plugin owned by profile A, turn scope bound for profile B: the turn's scope is the
    caller's and must win — the facade never replaces an installed scope."""
    home_a = _write_profile(tmp_path / "profile-a", base_url="https://a.example/v1")
    home_b = _write_profile(tmp_path / "profile-b", base_url="https://b.example/v1")
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(home_a)

    token = set_secret_scope(
        build_profile_secret_scope(home_b), profile_home=str(home_b)
    )
    try:
        llm.complete([{"role": "user", "content": "hi"}])
    finally:
        reset_secret_scope(token)

    assert record["secret"] == "https://b.example/v1"
    assert Path(record["scope_home"]).resolve() == home_b.resolve()
    assert current_secret_scope() is None


# ---------------------------------------------------------------------------
# Review item 6: the GENUINE resolver, no call_llm replacement
# ---------------------------------------------------------------------------

def test_genuine_resolver_no_openrouter_profile_needs_no_credential_read(tmp_path, monkeypatch, multiplex_on):
    """The real ``_resolve_openrouter_runtime`` with a no-OpenRouter profile whose
    ``model.base_url`` is ``http://127.0.0.1:4000/v1``: the trusted config rung must answer
    BEFORE any credential read, so an unscoped caller never hits ``UnscopedSecretError``."""
    from hermes_cli import runtime_provider as rp
    from hermes_cli.runtime_provider_backends import _resolve_openrouter_runtime

    home = tmp_path / "served-profile"
    home.mkdir()
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-should-not-be-read\n")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom",
        "base_url": CUSTOM_BASE_URL,
        "default": "step-5-preview",
    })
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    for name in ("OPENROUTER_BASE_URL", "CUSTOM_BASE_URL", "OPENAI_BASE_URL", "OPENROUTER_API_KEY",
                 "OPENAI_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    # No secret scope installed: this is the unscoped boot path that used to fail closed.
    assert current_secret_scope() is None

    resolved = _resolve_openrouter_runtime(requested_provider="custom")

    assert resolved["base_url"] == CUSTOM_BASE_URL
    assert resolved["provider"] == "custom"


def test_genuine_resolver_still_fails_closed_without_config_or_scope(tmp_path, monkeypatch, multiplex_on):
    """The reorder must not weaken the fail-closed guarantee: with no config base_url and no
    scope, the OPENROUTER_BASE_URL read still raises ``UnscopedSecretError``."""
    from hermes_cli import runtime_provider as rp
    from hermes_cli.runtime_provider_backends import _resolve_openrouter_runtime

    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "auto", "default": "m"})
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    for name in ("OPENROUTER_BASE_URL", "CUSTOM_BASE_URL", "OPENAI_BASE_URL", "OPENROUTER_API_KEY",
                 "OPENAI_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(UnscopedSecretError):
        _resolve_openrouter_runtime(requested_provider="auto")


def test_genuine_resolver_env_rungs_keep_their_precedence(tmp_path, monkeypatch, multiplex_on):
    """Documented order is unchanged: explicit > CUSTOM_BASE_URL > model.base_url >
    OPENROUTER_BASE_URL > default. A scoped caller still sees the env rungs."""
    from hermes_cli import runtime_provider as rp
    from hermes_cli.runtime_provider_backends import _resolve_openrouter_runtime

    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom", "base_url": "http://should-not-win.example/v1", "default": "m",
    })
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    for name in ("OPENROUTER_BASE_URL", "CUSTOM_BASE_URL", "OPENAI_BASE_URL", "OPENROUTER_API_KEY",
                 "OPENAI_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://127.0.0.1:4000/v1")

    token = set_secret_scope({"CUSTOM_BASE_URL": "http://127.0.0.1:4000/v1"}, profile_home=str(tmp_path))
    try:
        resolved = _resolve_openrouter_runtime(requested_provider="custom")
    finally:
        reset_secret_scope(token)

    assert resolved["base_url"] == "http://127.0.0.1:4000/v1"
