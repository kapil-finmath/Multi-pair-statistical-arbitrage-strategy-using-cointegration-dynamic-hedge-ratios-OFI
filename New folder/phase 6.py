# ==========================================================
# PHASE 6
# PROXY ORDER FLOW IMBALANCE (OFI)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

prices = pd.read_csv(
    "data/close_prices.csv",
    index_col=0,
    parse_dates=True
)

volume = pd.read_csv(
    "data/volume.csv",
    index_col=0,
    parse_dates=True
)

# ==========================================================
# BEST PAIR
# ==========================================================

asset1 = "IVV"
asset2 = "VGT"

# ==========================================================
# RETURNS
# ==========================================================

ret1 = np.log(prices[asset1]).diff()
ret2 = np.log(prices[asset2]).diff()

# ==========================================================
# SIGNED VOLUME OFI
# ==========================================================

ofi1 = volume[asset1] * np.sign(ret1)

ofi2 = volume[asset2] * np.sign(ret2)

# ==========================================================
# PAIR OFI
# ==========================================================

pair_ofi = ofi1 - ofi2

# ==========================================================
# OFI Z SCORE
# ==========================================================

window = 60

ofi_mean = pair_ofi.rolling(window).mean()
ofi_std  = pair_ofi.rolling(window).std()

ofi_z = (pair_ofi - ofi_mean) / ofi_std

# ==========================================================
# SAVE RESULTS
# ==========================================================

results = pd.DataFrame({
    "OFI": pair_ofi,
    "OFI_ZScore": ofi_z
})

results.to_csv(
    "results/pair_ofi.csv"
)

# ==========================================================
# PLOT OFI
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(pair_ofi)

plt.title(
    "Pair Order Flow Imbalance (IVV - VGT)"
)

plt.tight_layout()

plt.savefig(
    "figures/pair_ofi.png"
)

plt.close()

# ==========================================================
# PLOT OFI Z SCORE
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(ofi_z)

plt.axhline(2, linestyle="--")
plt.axhline(-2, linestyle="--")
plt.axhline(0)

plt.title("OFI Z Score")

plt.tight_layout()

plt.savefig(
    "figures/ofi_zscore.png"
)

plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

print("="*60)
print("PAIR OFI SUMMARY")
print("="*60)

print()
print("Mean OFI :", pair_ofi.mean())
print("Std OFI  :", pair_ofi.std())

print()
print("Max OFI  :", pair_ofi.max())
print("Min OFI  :", pair_ofi.min())

print()
print("="*60)
print("PHASE 6 COMPLETE")
print("="*60)

print()
print("Generated:")
print("results/pair_ofi.csv")
print("figures/pair_ofi.png")
print("figures/ofi_zscore.png")
