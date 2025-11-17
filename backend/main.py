from fastapi import FastAPI

from .routers import items

app = FastAPI(title="PsoMetric Backend")

app.include_router(items.router)


@app.get("/")
def read_root():
    return {"status": "ok", "service": "PsoMetric Backend"}
