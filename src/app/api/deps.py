"""Dependency providers. Tests swap any of them with app.dependency_overrides."""

from typing import Annotated

from fastapi import Depends, Request

from app.controllers.search import SearchController
from app.core.config import Settings
from app.core.resources import Resources
from app.core.security import app_settings
from app.services.lens import Lens


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


def get_lens(resources: Annotated[Resources, Depends(get_resources)]) -> Lens:
    return resources.lens


def get_search_controller(
    lens: Annotated[Lens, Depends(get_lens)], settings: Annotated[Settings, Depends(app_settings)]
) -> SearchController:
    return SearchController(lens, settings)
