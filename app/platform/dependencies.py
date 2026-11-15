"""Dependency providers shared by every feature. Tests swap any of them with app.dependency_overrides."""

from typing import Annotated

from fastapi import Depends, Request

from app.platform.errors import RateLimited
from app.platform.rate_limit import DAY_SECONDS, RateLimiter, caller_of
from app.platform.resources import Resources
from config.settings import Settings


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


def get_settings(request: Request) -> Settings:
    """The settings this app instance was built with (not a global), so tests and workers can differ."""
    settings: Settings = request.app.state.settings
    return settings


def get_rate_limiter(resources: Annotated[Resources, Depends(get_resources)]) -> RateLimiter:
    return resources.rate_limiter


async def _check_cap(
    rate_limiter: RateLimiter, caller: str, limit: int, window_seconds: int, prefix: str, message: str
) -> None:
    if limit <= 0:
        return
    retry_after = await rate_limiter.hit(caller, limit, window_seconds=window_seconds, prefix=prefix)
    if retry_after is not None:
        raise RateLimited(message, retry_after)


async def enforce_rate_limit(
    request: Request,
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    caller = caller_of(request, settings.api_key_list)
    await _check_cap(
        rate_limiter,
        caller,
        settings.rate_limit_per_minute,
        60,
        "rate",
        f"Over {settings.rate_limit_per_minute} requests a minute.",
    )
    await _check_cap(
        rate_limiter,
        caller,
        settings.daily_requests_per_caller,
        DAY_SECONDS,
        "daily_caller",
        f"Daily limit reached ({settings.daily_requests_per_caller} requests per caller). Try again tomorrow.",
    )
    await _check_cap(
        rate_limiter,
        "all",
        settings.daily_requests_total,
        DAY_SECONDS,
        "daily_total",
        "The public daily request cap has been reached. Try again tomorrow.",
    )


async def enforce_upload_limit(
    request: Request,
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    caller = caller_of(request, settings.api_key_list)
    await _check_cap(
        rate_limiter,
        caller,
        settings.daily_uploads_per_caller,
        DAY_SECONDS,
        "daily_upload_caller",
        f"Daily upload limit reached ({settings.daily_uploads_per_caller} uploads per caller). Try again tomorrow.",
    )
    await _check_cap(
        rate_limiter,
        "all",
        settings.daily_uploads_total,
        DAY_SECONDS,
        "daily_upload_total",
        "The public daily upload cap has been reached. Try again tomorrow.",
    )
