
from datasets import load_dataset
import time

def main():
    print("Downloading MATH dataset manually...")
    # MATH has multiple configs (subjects). We should load them all or just the ones needed.
    # The loader in benchmarks.py loads specific subjects.
    # Let's try loading the main one first.
    
    subsets = [
        "algebra", "counting_and_probability", "geometry", "intermediate_algebra", 
        "number_theory", "prealgebra", "precalculus"
    ]
    
    for subj in subsets:
        print(f"Downloading subset: {subj} ...")
        try:
            load_dataset("EleutherAI/hendrycks_math", subj, split="test")
            print(f"✓ {subj} cached.")
        except Exception as e:
            print(f"✗ Failed {subj}: {e}")
            
    print("All downloads attempted.")

if __name__ == "__main__":
    main()
