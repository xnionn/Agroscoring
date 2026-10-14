import os
import sys

# Ensure platform dir is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from scoring import scoring_service
from seed_data import seed_demo_data

from routers.auth_router import router as auth_router
from routers.farmers import router as farmers_router
from routers.applications import router as applications_router
from routers.analytics import router as analytics_router

app = FastAPI(title="АгроСкоринг", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(farmers_router)
app.include_router(applications_router)
app.include_router(analytics_router)

# Static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup():
    init_db()
    scoring_service.load()
    seed_demo_data()


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/auth")
def auth_page():
    return FileResponse(os.path.join(STATIC_DIR, "auth.html"))


@app.get("/portal")
def portal_page():
    return FileResponse(os.path.join(STATIC_DIR, "portal.html"))


@app.get("/dashboard")
def dashboard_page():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))
