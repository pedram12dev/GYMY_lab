from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.interfaces.api import router as api_router
from app.interfaces.ws_routes import router as ws_router

settings = get_settings()
app = FastAPI(title="Gymy Smart")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health(): return {"status": "ok"}

app.include_router(api_router, prefix="/api")
app.include_router(ws_router)
