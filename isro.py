import lightkurve as lk
import matplotlib.pyplot as plt

# 1. Download light curve data from NASA's Kepler mission (Example: Kepler-8)
print("Downloading light curve data...")
search_result = lk.search_lightcurve('Kepler-8', author='Kepler', quarter=1)
lc = search_result.download()

# 2. Remove NaN (missing) values and outliers
lc = lc.remove_nans().remove_outliers()

# 3. Separate starspots from the planet (Detrending)
# We use a Savitzky-Golay filter to smooth out long-term starspot waves
# window_length: Number of data points for the smoothing window (must be odd)
flattened_lc, trend_lc = lc.flatten(window_length=401, return_trend=True)

# 4. Search for the planet's orbital period
# The Periodogram uses a Box Least Squares (BLS) algorithm to find transit shapes
periodogram = flattened_lc.to_periodogram(method='bls')
best_period = periodogram.period_at_max_power

print(f"Detected Planet Period: {best_period:.4f}")

# 5. Plot the results to visualize the difference
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

# Plot 1: Raw Data showing starspot waves and transit dips combined
lc.scatter(ax=axes[0], color='black', size=1, label='Raw Data (Spots + Planet)')
trend_lc.plot(ax=axes[0], color='red', linewidth=2, label='Starspot Trend Line')
axes[0].set_title("Raw Light Curve with Starspot Variations")

# Plot 2: Cleaned Data (The starspots are flattened out, leaving just the transits)
flattened_lc.scatter(ax=axes[1], color='blue', size=1, label='Detrended Data')
axes[1].set_title("Cleaned Light Curve (Starspots Removed)")

# Plot 3: Folded Data (Stacking all transits on top of each other to see the planet shape)
folded_lc = flattened_lc.fold(period=best_period)
folded_lc.scatter(ax=axes[2], color='purple', size=2, label='Folded Transit')
axes[2].set_title(f"Folded Light Curve at Period: {best_period:.4f}")

plt.tight_layout()
plt.show()