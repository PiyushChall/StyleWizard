from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
from tinydb import TinyDB, Query
from pathlib import Path
import mediapipe as mp
from contextlib import asynccontextmanager

# TinyDB setup
DATABASE_PATH = "filters.json"
db = TinyDB(DATABASE_PATH)

# Load face detection and landmark tracking models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create the "uploads" directory if it doesn't exist
    Path("uploads").mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: No specific shutdown tasks in this case

# Initialize FastAPI app with lifespan handler
app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/admin")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    filters = db.all()
    return templates.TemplateResponse("index.html", {"request": request, "filters": filters})

@app.post("/upload")
async def upload_filter(request: Request, name: str = Form(...), file: UploadFile = File(...)):
    try:
        # Save the uploaded image
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Store filter data in TinyDB
        db.insert({"name": name, "image_path": file_path})

        return RedirectResponse(url="/")
    except Exception as e:
        return {"error": str(e)}

@app.get("/delete/{filter_id}")
async def delete_filter(request: Request, filter_id: int):
    try:
        # Delete the filter from TinyDB
        db.remove(doc_ids=[filter_id])

        return RedirectResponse(url="/")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/filters")
async def get_filters():
    filters = db.all()
    return {"filters": filters}