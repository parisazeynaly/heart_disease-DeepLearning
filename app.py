from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    """Check if API is running."""
    return {"message": "FastAPI is running!"}

# Run FastAPI when executing app.py (Without `reload=True`)
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
