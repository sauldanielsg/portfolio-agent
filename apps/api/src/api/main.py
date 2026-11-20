from fastapi import FastAPI

from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from api.routes.profile import router as profile_router

app = FastAPI(title="AI Portfolio API")
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(profile_router)
