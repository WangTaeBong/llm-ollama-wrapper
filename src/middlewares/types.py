from typing import Optional, Dict, Any

from fastapi import FastAPI


class ApplicationState:
    """
    Defines shared state for the application.

    This class provides type-safe access to global resources that need to be
    accessible throughout the application lifecycle, such as database connections,
    cache clients, and machine learning models.

    Attributes:
        sync_redis_client (Optional[Any]): Synchronous Redis client instance.
            Used for non-async code paths requiring Redis access.

        async_redis_client (Optional[Any]): Asynchronous Redis client instance.
            Used in async code paths for non-blocking Redis operations.

        llm_models (Dict[str, Any]): Dictionary of loaded language models,
            where keys are model identifiers and values are model instances.
            Allows efficient access to pre-loaded models without reinitialization.
    """
    sync_redis_client: Optional[Any] = None
    async_redis_client: Optional[Any] = None
    llm_models: Dict[str, Any] = {}


class AppWithState(FastAPI):
    """
    FastAPI extension class with explicitly typed state.

    Extends the standard FastAPI class to provide type-safe access to application state,
    improving IDE auto-completion and enabling static type checking for state attributes.

    This approach avoids the need for type casting when accessing state properties
    and provides better code documentation for shared resources.

    Attributes:
        state (ApplicationState): Strongly-typed application state container.
            Replaces the default generic state object with a specifically typed instance.
    """
    state: ApplicationState
