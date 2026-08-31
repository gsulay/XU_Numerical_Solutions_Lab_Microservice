from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from activities.highway_alignment import get_spiral_standards

app = FastAPI(
    title="Numerical Solutions Laboratory Activities API", 
    description="Backend microservices for computational engineering laboratories."
)

# Optional: Add CORS middleware if your notebooks are hosted on a different domain 
# and you ever transition from Python requests to browser-based JavaScript fetch calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"title":"Numerical Solutions Laboratory Activities API", 
        "description":"Backend microservices for computational engineering laboratories."}

@app.get("/api/v1/alignment/standards/{alignment_id}")
def read_alignment_standards(alignment_id: str):
    """
    Retrieves the geometric constraints for a specific highway alignment assignment.
    """
    try:
        data = get_spiral_standards(alignment_id)
        return data
    except ValueError as e:
        # Catch our custom format error and return a clean 400 Bad Request
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected mathematical or processing errors
        raise HTTPException(status_code=500, detail="Internal server processing error.")

# Example placeholder for future lab modules:
# @app.post("/api/v1/structures/truss_analysis")
# def analyze_truss(data: dict): ...