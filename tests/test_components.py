import pytest
import pandas as pd
from pydantic import ValidationError

# Import the validators we just built
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_validators import DrugRecord, InteractionRecord, clean_and_validate_dataframe

def test_drug_record_valid():
    """Test 1: Prevents Context Loss by testing valid schema adherence"""
    record = DrugRecord(drugbank_id="DB123", name="Aspirin", all_names="Aspirin|Bayer")
    assert record.name == "Aspirin"

def test_drug_record_invalid_garbage_out():
    """Test 2: Prevents Garbage In, Garbage Out by catching malformed IDs"""
    with pytest.raises(ValidationError):
        # drugbank_id requires min_length=3, so "D" drops the record
        DrugRecord(drugbank_id="D", name="BadDrug")

def test_dataframe_cleaning():
    """Test 3: Anti-Compounding Errors. Pipeline must survive dirty data injections without crashing"""
    raw_dirty_df = pd.DataFrame([
        {"drugbank_id": "DB1234", "name": "ValidDrug", "all_names": ""},
        {"drugbank_id": "XY", "name": "CorruptedData", "all_names": "LostContext"},  # Should be Dropped
        {"drugbank_id": "DB999", "name": "", "all_names": None} # Missing Name => Dropped
    ])
    
    clean_df = clean_and_validate_dataframe(raw_dirty_df, DrugRecord)
    
    # We injected 3 rows, 2 are garbage, so only 1 should survive the enterprise vacuum.
    assert len(clean_df) == 1
    assert clean_df.iloc[0]['name'] == 'ValidDrug'
