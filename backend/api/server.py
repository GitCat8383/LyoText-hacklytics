"""FastAPI application factory with CORS and lifespan management."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.websocket import ws_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    ws_manager.set_loop(loop)

    if hasattr(app.state, "orchestrator"):
        app.state.orchestrator.start()

    logger.info("FastAPI server started")
    yield
    logger.info("FastAPI server shutting down")

    if hasattr(app.state, "orchestrator"):
        app.state.orchestrator.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hacklytic BCI",
        description="Hybrid Brain-Computer Interface for Assistive Communication",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.websocket("/ws/events")
    async def websocket_endpoint(websocket):
        await ws_manager.handle(websocket)

    return app
