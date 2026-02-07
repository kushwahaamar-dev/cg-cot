
import pandas as pd
from src.analysis import full_analysis, RESULTS_DIR

def test():
    df = pd.read_csv(RESULTS_DIR / "results_dummy.csv")
    full_analysis(df)

if __name__ == "__main__":
    test()
