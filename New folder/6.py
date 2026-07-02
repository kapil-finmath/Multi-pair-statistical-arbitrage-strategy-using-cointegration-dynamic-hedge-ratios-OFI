import pandas as pd
import numpy as np

spread = pd.read_csv(
    "results/spread.csv",
    index_col=0
)

spread = spread.squeeze()

print("Static Spread Variance")
print(np.var(spread))
