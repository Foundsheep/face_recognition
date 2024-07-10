from fastapi import FastAPI
from scripts import save_frames

app = FastAPI()

@app.get("/")
def hello_world():
    return {"message": "OK"}


