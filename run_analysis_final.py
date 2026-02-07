
import pandas as pd
from src.analysis import full_analysis, load_results

def run():
    print("Loading results...")
    try:
        # Load whatever we have, even if partial
        df = load_results(matched=False)
        print(f"Loaded {len(df)} rows.")
        
        # Run full analysis pipeline
        full_analysis(df)
        print("Analysis complete.")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    run()
