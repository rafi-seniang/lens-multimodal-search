"""How the search feature is reached per request: Lens built at startup, then a controller around it."""

from typing import Annotated

from fastapi import Depends

from app.features.search.controller import SearchController
from app.features.search.service import Lens
from app.platform.dependencies import get_resources, get_settings
from app.platform.resources import Resources
from config.settings import Settings


def get_lens(resources: Annotated[Resources, Depends(get_resources)]) -> Lens:
    return resources.lens


def get_search_controller(
    lens: Annotated[Lens, Depends(get_lens)], settings: Annotated[Settings, Depends(get_settings)]
) -> SearchController:
    return SearchController(lens, settings)
