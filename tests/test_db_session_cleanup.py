"""
tests/test_db_session_cleanup.py
==================================
Regression test for the Phase 2 incident.

`get_db()` in app/database.py is a FastAPI dependency written with
`yield`. FastAPI drives dependencies like this the same way a context
manager works: it runs up to `yield`, calls the endpoint, and if the
endpoint raises (including an intentional `HTTPException` for a 404 or
422), that exception is thrown back into the generator at the point of
`yield`. Only code wrapped in `try`/`finally` around the `yield` will
still run in that case.

`get_db()` no longer wraps `yield` in `try`/`finally`, so `db.close()`
is skipped whenever the request fails for any reason -- including
completely ordinary, expected failures like "order not found" (404) or
"invalid payload" (422). Each one leaks a connection back to the pool.

This test does not require Postgres or Docker: it drives the
dependency generator directly, the same way FastAPI's dependency
injection does internally, and checks whether the session was closed.
It intentionally fails against the current app/database.py.
"""

from unittest.mock import patch

import pytest

from app.database import get_db


def test_get_db_closes_session_even_when_the_request_raises():
    """
    A correctly implemented `get_db` always returns its connection to
    the pool, whether the request succeeded or failed. Without a
    try/finally around `yield`, every failed request leaks one.
    """
    gen = get_db()
    db = next(gen)

    with patch.object(db, "close", wraps=db.close) as close_spy:
        with pytest.raises(RuntimeError):
            gen.throw(RuntimeError("simulated downstream failure"))

        close_spy.assert_called_once()
