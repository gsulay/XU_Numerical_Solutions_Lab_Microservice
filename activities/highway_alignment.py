import random

def get_spiral_standards(alignment_id: str) -> dict:
    """
    Uses the entire alignment_id string as a deterministic seed 
    to generate highway geometry parameters and ROW limits.
    """

    MIN_SAFE_ID = "SAYRE_HWY_9999"
    MAX_SAFE_ID = "SAYRE_HWY_1111"

    # Just ensure they didn't send a completely blank string
    cleaned_id = alignment_id.strip()
    if not cleaned_id:
        raise ValueError("Alignment ID cannot be empty.")
    
    # Instantiate a local random generator using the entire string.
    # Python natively converts strings to a deterministic hash for seeding.
    rng = random.Random(cleaned_id)
    
    # Generate parameters within typical highway design bounds
    R_c = round(rng.uniform(200.0, 500.0), 1)
    L_s = round(rng.uniform(50.0, 150.0), 1)
    
    # Calculate the estimated offset: y ≈ (L_s^2) / (6 * R_c)
    estimated_y = (L_s**2) / (6 * R_c)
    
    # Generate a tight ROW limit multiplier (0.7 to 1.5) to ensure a mix of pass/fail
    row_multiplier = rng.uniform(0.7, 1.5)
    max_row = round(estimated_y * row_multiplier, 2)
    
    # Enforce a minimum practical limit
    max_row = max(max_row, 1.5)

    #Safe ID for checking
    if alignment_id == MIN_SAFE_ID:
        max_row = round(estimated_y * 0.5, 2)
            
        # Enforce a minimum practical limit
        max_row = max(max_row, 1.5)
    elif alignment_id == MAX_SAFE_ID:
        max_row = round(estimated_y * 1.5, 2)
                    
        # Enforce a minimum practical limit
        max_row = max(max_row, 1.5)
         
    
    return {
        "R_c": R_c,
        "L_s": L_s,
        "max_allowable_row_offset": max_row
    }