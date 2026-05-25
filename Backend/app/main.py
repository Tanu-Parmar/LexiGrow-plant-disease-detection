from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="LexiGrow API")

app.include_router(router)