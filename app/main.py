from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services import ai_engine
from app.routers import analyze

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load heavy models once
    ai_engine.load_models()
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Psoriasis AI Backend",
    version="1.0", 
    lifespan=lifespan
)

# CORS (Important for React Native connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Router
app.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])

@app.get("/")
def root():
    return {"status": "Online", "message": "Psoriasis AI Backend Running"}