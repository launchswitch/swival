"""Tests for swival.keepawake context manager behavior."""

import sys
import threading

import pytest

import swival.keepawake as keepawake
from swival.keepawake import keep_awake


@pytest.fixture(autouse=True)
def _reset_keepawake_state():
    keepawake._LOAD_FAILED = False
    keepawake._IOKIT = None
    keepawake._CFD = None
    keepawake._REFCOUNT = 0
    keepawake._ASSERTION_ID = None


@pytest.fixture
def assertion_calls(monkeypatch):
    """Monkeypatch _create/_release; return (created reasons, released ids)."""
    created: list[str] = []
    released: list[int] = []

    def fake_create(reason: str) -> int:
        created.append(reason)
        return 123

    monkeypatch.setattr(keepawake, "_create", fake_create)
    monkeypatch.setattr(keepawake, "_release", released.append)
    return created, released


def test_keep_awake_single_call(assertion_calls):
    created, released = assertion_calls

    with keep_awake():
        assert created == ["swival agent turn"]
        assert released == []

    assert created == ["swival agent turn"]
    assert released == [123]


def test_keep_awake_nested_calls_share_single_assertion(assertion_calls):
    created, released = assertion_calls

    with keep_awake(reason="nested"):
        with keep_awake(reason="nested"):
            assert created == ["nested"]
            assert released == []

    assert created == ["nested"]
    assert released == [123]


def test_keep_awake_cross_thread_overlap(assertion_calls):
    created, released = assertion_calls
    a_entered = threading.Event()
    a_exit = threading.Event()
    b_entered = threading.Event()

    def thread_a():
        with keep_awake():
            a_entered.set()
            a_exit.wait(timeout=2)

    def thread_b():
        with keep_awake():
            b_entered.set()

    ta = threading.Thread(target=thread_a)
    ta.start()
    assert a_entered.wait(timeout=2)

    tb = threading.Thread(target=thread_b)
    tb.start()
    assert b_entered.wait(timeout=2)

    # Keep A alive while B enters and exits; still no release yet.
    tb.join(timeout=2)
    assert not tb.is_alive()
    assert len(created) == 1
    assert released == []

    a_exit.set()
    ta.join(timeout=2)

    assert not ta.is_alive()
    assert len(created) == 1
    assert released == [123]


def test_keep_awake_creator_exits_first(assertion_calls):
    created, released = assertion_calls
    a_entered = threading.Event()
    a_exited = threading.Event()
    b_entered = threading.Event()
    b_release = threading.Event()

    def thread_a():
        with keep_awake():
            a_entered.set()
            b_entered.wait(timeout=2)
        a_exited.set()

    def thread_b():
        assert a_entered.wait(timeout=2)
        with keep_awake():
            b_entered.set()
            b_release.wait(timeout=2)

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()

    # A (the creator) exits while B still holds the context: no release yet.
    assert a_exited.wait(timeout=2)
    assert len(created) == 1
    assert released == []

    b_release.set()
    ta.join(timeout=2)
    tb.join(timeout=2)

    assert not ta.is_alive()
    assert not tb.is_alive()
    assert len(created) == 1
    assert released == [123]


def test_keep_awake_releases_on_exception(assertion_calls):
    created, released = assertion_calls

    with pytest.raises(RuntimeError, match="boom"):
        with keep_awake():
            raise RuntimeError("boom")

    assert len(created) == 1
    assert released == [123]


def test_keep_awake_no_release_if_create_fails(assertion_calls, monkeypatch):
    _, released = assertion_calls
    monkeypatch.setattr(keepawake, "_create", lambda reason: None)

    with keep_awake():
        pass

    assert released == []


def test_keep_awake_as_decorator_recreates_every_call(assertion_calls):
    created, released = assertion_calls

    @keep_awake()
    def call_me():
        return True

    call_me()
    call_me()

    assert len(created) == 2
    assert released == [123, 123]


def test_keep_awake_is_noop_on_non_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    released: list[int] = []
    monkeypatch.setattr(keepawake, "_release", released.append)

    with keep_awake():
        assert keepawake._ASSERTION_ID is None

    assert keepawake._LOAD_FAILED
    assert released == []
