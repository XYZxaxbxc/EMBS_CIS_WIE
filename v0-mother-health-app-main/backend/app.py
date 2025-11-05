from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import food

app = FastAPI(
    title="Mother Health API",
    version="0.1.0",
    description="Backend services for the Mother Health platform.",
)

# Allow the Next.js client and local development tools to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prefix all backend endpoints with /api to match the frontend proxy.
app.include_router(food.router, prefix="/api", tags=["nutrition"])


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Lightweight readiness probe used by local dev scripts."""
    return {"status": "ok"}
