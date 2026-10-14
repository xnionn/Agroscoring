"""
Data Profiler — analyze available datasets in Downloads to understand their structure
for integration into the scoring system.
"""
import pandas as pd
import json
import os
from pathlib import Path

DL = r"C:\Users\Мади\Downloads"

def profile_excel_files():
    """Profile all relevant Excel files."""
    excel_files = [
        "Диаграмма 1 рус.xlsx",
        "Т-03-02-М 08 2025 рус.xlsx",
        "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx",
    ]
    
    for fname in excel_files:
        path = os.path.join(DL, fname)
        if not os.path.exists(path):
            print(f"\n✗ {fname} — NOT FOUND")
            continue
        
        try:
            xls = pd.ExcelFile(path)
            print(f"\n{'='*60}")
            print(f"FILE: {fname}")
            print(f"{'='*60}")
            print(f"Sheets: {xls.sheet_names}")
            
            for sheet in xls.sheet_names[:2]:  # First 2 sheets
                df = pd.read_excel(path, sheet_name=sheet)
                print(f"\n  SHEET '{sheet}':")
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)[:10]}")
                if len(df) > 0:
                    print(f"    Sample values:\n{df.iloc[0].to_string()[:500]}")
        except Exception as e:
            print(f"\n✗ {fname} — ERROR: {e}")

def profile_csv_files():
    """Profile CSV metadata files."""
    csv_pattern = [
        "--Granularity-Joinkeys.csv",
        "datasetname-countryscope-sourceurl-sourcedomain-pu.csv",
    ]
    
    for fname in csv_pattern:
        path = os.path.join(DL, fname)
        if not os.path.exists(path):
            print(f"\n✗ {fname} — NOT FOUND")
            continue
        
        try:
            df = pd.read_csv(path)
            print(f"\n{'='*60}")
            print(f"FILE: {fname}")
            print(f"{'='*60}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 3 rows:\n{df.head(3)}")
        except Exception as e:
            print(f"\n✗ {fname} — ERROR: {e}")

def profile_json_files():
    """Profile JSON data files."""
    json_pattern = [
        "organikalyk_onim_ondirushilerd-v4.json",
        "reestr_uboihraneniepererab-v10.json",
        "t361-v8.json",
    ]
    
    for fname in json_pattern:
        path = os.path.join(DL, fname)
        if not os.path.exists(path):
            print(f"\n✗ {fname} — NOT FOUND")
            continue
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n{'='*60}")
            print(f"FILE: {fname}")
            print(f"{'='*60}")
            
            if isinstance(data, list):
                print(f"Type: Array with {len(data)} items")
                if data:
                    print(f"First item keys: {list(data[0].keys())[:10] if isinstance(data[0], dict) else 'non-dict'}")
                    print(f"Sample: {str(data[0])[:300]}")
            elif isinstance(data, dict):
                print(f"Type: Dict with keys: {list(data.keys())[:10]}")
                print(f"Sample: {str(data)[:300]}")
        except Exception as e:
            print(f"\n✗ {fname} — ERROR: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATA PROFILING FOR SCORING SYSTEM")
    print("="*60)
    
    profile_excel_files()
    profile_csv_files()
    profile_json_files()
    
    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)
