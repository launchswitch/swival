"""macOS power-management assertion helper.

Provides a context manager that acquires an
`IOPMAssertionCreateWithName(PreventUserIdleSystemSleep)` while active and
releases it when the context exits. This keeps macOS from entering
idle-sleep while agent turns are running.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from contextlib import contextmanager


_ASSERTION_TYPE = b"PreventUserIdleSystemSleep"
_ASSERTION_LEVEL = 255  # kIOPMAssertionLevelOn
_CF_STRING_ENCODING_UTF8 = 0x08000100

_LOCK = threading.Lock()
_REFCOUNT = 0
_ASSERTION_ID: int | None = None
_LOAD_FAILED = False

_IOKIT = None
_CFD = None


def _load() -> bool:
    """Load and cache the required IOKit/CoreFoundation libraries."""
    global _LOAD_FAILED, _IOKIT, _CFD

    if _LOAD_FAILED:
        return False
    if _IOKIT is not None:
        return True
    if sys.platform != "darwin":
        _LOAD_FAILED = True
        return False

    try:
        iokit = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/IOKit.framework/IOKit"
        )
        cfd = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )

        iokit.IOPMAssertionCreateWithName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        iokit.IOPMAssertionCreateWithName.restype = ctypes.c_int32

        iokit.IOPMAssertionRelease.argtypes = [ctypes.c_uint32]
        iokit.IOPMAssertionRelease.restype = ctypes.c_int32

        cfd.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        cfd.CFStringCreateWithCString.restype = ctypes.c_void_p

        cfd.CFRelease.argtypes = [ctypes.c_void_p]
        cfd.CFRelease.restype = None
    except Exception:
        _LOAD_FAILED = True
        return False

    _IOKIT = iokit
    _CFD = cfd
    return True


def _create(reason: str) -> int | None:
    """Create an assertion and return its assertion ID, or ``None`` on failure."""
    if not _load():
        return None

    assertion_type = _CFD.CFStringCreateWithCString(
        None, _ASSERTION_TYPE, _CF_STRING_ENCODING_UTF8
    )
    if not assertion_type:
        return None

    reason_for_activity = _CFD.CFStringCreateWithCString(
        None, reason.encode("utf-8"), _CF_STRING_ENCODING_UTF8
    )
    if not reason_for_activity:
        _CFD.CFRelease(assertion_type)
        return None

    assertion_id = ctypes.c_uint32(0)
    try:
        rv = _IOKIT.IOPMAssertionCreateWithName(
            assertion_type,
            _ASSERTION_LEVEL,
            reason_for_activity,
            ctypes.byref(assertion_id),
        )
    except Exception:
        return None
    finally:
        _CFD.CFRelease(assertion_type)
        _CFD.CFRelease(reason_for_activity)

    if rv != 0:
        return None

    return int(assertion_id.value)


def _release(assertion_id: int) -> None:
    """Release an active assertion by ID."""
    if _load():
        _IOKIT.IOPMAssertionRelease(assertion_id)


@contextmanager
def keep_awake(*, reason: str = "swival agent turn"):
    """Keep macOS from entering idle-sleep while inside the context.

    Nested and concurrent contexts are supported via process-wide refcounting.
    """
    global _REFCOUNT, _ASSERTION_ID

    with _LOCK:
        if _REFCOUNT == 0:
            created_id = _create(reason)
            if created_id is not None:
                _ASSERTION_ID = created_id
        _REFCOUNT += 1

    try:
        yield
    finally:
        release_id = None
        with _LOCK:
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT == 0 and _ASSERTION_ID is not None:
                release_id = _ASSERTION_ID
                _ASSERTION_ID = None

        if release_id is not None:
            _release(release_id)
