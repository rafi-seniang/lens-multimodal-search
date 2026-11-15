"""Who the rate limit counts as one caller."""

from starlette.requests import Request

from app.platform import rate_limit


def _request(headers: dict[str, str]) -> Request:
    raw = [(name.lower().encode(), value.encode()) for name, value in headers.items()]
    return Request({"type": "http", "headers": raw, "client": ("198.51.100.7", 5000)})


def test_only_an_accepted_key_is_a_caller_of_its_own() -> None:
    assert rate_limit.caller_of(_request({"X-API-Key": "test-key"}), ["test-key"]) == "key:test-key"
    # An invented key would otherwise start a fresh count on every request
    assert rate_limit.caller_of(_request({"X-API-Key": "made-up"}), ["test-key"]) == "ip:198.51.100.7"
    assert rate_limit.caller_of(_request({"X-API-Key": "made-up"}), []) == "ip:198.51.100.7"
