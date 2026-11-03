from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from activities.highway_alignment import get_spiral_standards
from activities import cramers_rule_truss, jacobi_water_network, gauss_seidel_traffic
from activities import greenshields_traffic, beam_polynomial, concrete_regression

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

@app.get("/api/v1/la2/cramers")
def get_cramers_rule_params(id: str = Query(..., description="The Student's unique ID")):
    """
    Returns the A_matrix and b_vector for the Cramer's Rule Determinate Truss lab.
    """
    # Fetch parameters from the activity module
    params = cramers_rule_truss.generate_params(id)
    return params

@app.get("/api/v1/la2/water_network")
def get_water_network_params(id: str = Query(..., description="The Student's unique ID")):
    """
    Returns the shuffled A_matrix and b_vector for the Jacobi Iteration lab.
    The student must perform row-swapping to achieve diagonal dominance.
    """
    # Fetch parameters from the activity module
    params = jacobi_water_network.generate_params(id)
    return params

@app.get("/api/v1/la2/traffic_network")
def get_traffic_network_params(id: str = Query(..., description="The Student's unique ID")):
    """
    Returns the A_matrix and b_vector for the Gauss-Seidel Traffic Flow lab.
    Ensures the matrix is strictly diagonally dominant.
    """
    # Fetch parameters from the activity module
    params = gauss_seidel_traffic.generate_params(id)
    return params

@app.get("/api/v1/la4/traffic_flow")
def get_traffic_flow_params(id: str = Query(..., description="The Student's unique ID")):
    """
    Returns speed-density readings and a query density for the LA4 linear regression scenario.
    """
    try:
        return greenshields_traffic.generate_params(id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server processing error.")

@app.get("/api/v1/la4/beam_moment")
def get_beam_moment_params(id: str = Query(..., description="The Student's unique ID")):
    """
    Returns beam gauge readings and a prediction point for the LA4 polynomial regression scenario.
    """
    try:
        return beam_polynomial.generate_params(id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server processing error.")

@app.get("/api/v1/la4/concrete_capacity")
def get_concrete_capacity_params(id: str = Query(..., description="The Student's unique ID")):
    """
    Returns column test data and a new column for the LA4 gradient descent scenario.
    """
    try:
        return concrete_regression.generate_params(id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server processing error.")

# Example placeholder for future lab modules:
# @app.post("/api/v1/structures/truss_analysis")
# def analyze_truss(data: dict): ...