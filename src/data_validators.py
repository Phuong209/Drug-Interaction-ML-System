from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Optional, List
import pandas as pd
import logging

# Configure robust enterprise logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataQualityGate")

class DrugRecord(BaseModel):
    """Schema validation for raw Drug Entities. Any bad rows (missing ID/Name) are rejected."""
    model_config = ConfigDict(coerce_numbers_to_str=True)
    
    drugbank_id: str = Field(..., min_length=3)
    name: str = Field(..., min_length=1)
    # Allows None/NaN natively translated to None
    all_names: Optional[str] = None 

class InteractionRecord(BaseModel):
    """Schema validation for Drug-Drug Interaction matrix."""
    model_config = ConfigDict(coerce_numbers_to_str=True)
    
    drug_1_id: str = Field(..., min_length=3)
    drug_1_name: str
    drug_2_id: str = Field(..., min_length=3)
    drug_2_name: str
    description: str = Field(..., min_length=5)

def clean_and_validate_dataframe(df: pd.DataFrame, ModelClass) -> pd.DataFrame:
    """
    Enterprise Data Quality Gate:
    1. Removes completely null empty rows.
    2. Drops exact duplicates.
    3. Runs each row through Pydantic strict typing validation.
    4. Drops 'Garbage In' rows that fail required schemas instead of crippling the system.
    """
    df = df.dropna(how='all').drop_duplicates()
    
    valid_rows = []
    dropped_count = 0
    
    # Fill NaN with None so Pydantic's Optional[] triggers correctly instead of receiving float(NaN)
    df = df.where(pd.notnull(df), None)
    
    for idx, row in df.iterrows():
        try:
            validated = ModelClass(**row.to_dict())
            valid_rows.append(validated.model_dump())
        except ValidationError as e:
            dropped_count += 1
            if dropped_count <= 5:  # Log only first few to prevent log flooding
                logger.warning(f"Data Quality Rejection at row {idx}. Reason: {e.errors()[0]['msg']}")
                
    if dropped_count > 0:
        logger.error(f"FATAL WARNING: Dropped {dropped_count} corrupted rows matching Garbage-In metrics!")
        
    return pd.DataFrame(valid_rows)
