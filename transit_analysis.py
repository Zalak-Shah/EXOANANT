import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import batman

# ==========================================
# 1. LOAD YOUR OWN CSV DATA
# ==========================================
# Replace 'your_data.csv' with your actual filename
csv_filename = 'info.csv' 

# Read the file using pandas
df = pd.read_csv(csv_filename)
print(len(df))


# Extract time and flux columns (change names if your CSV headers are different)
time = df['TIME'].values
flux = df['PDCSAP_FLUX'].values

# Convert into a Lightkurve object so we can use its advanced tools
lc = lk.LightCurve(time=time, flux=flux)
lc = lc.remove_nans().remove_outliers()

# ==========================================
# 2. DETREND (REMOVE STARSPOTS)
# ==========================================
# window_length must be an odd integer. Adjust it based on your data.
# A smaller window removes faster starspot changes.
flattened_lc, trend_lc = lc.flatten(window_length=101, return_trend=True)

# Find the best period using Box Least Squares (BLS)
periodogram = flattened_lc.to_periodogram(
    method='bls',
    minimum_period=0.5,
    maximum_period=20,
    frequency_factor=500
)
best_period = periodogram.period_at_max_power
t0 = periodogram.transit_time_at_max_power
duration = periodogram.duration_at_max_power

print(f"Detected Period: {best_period.value:.4f} days")
print(f"Detected Mid-Transit Time (t0): {t0.value:.4f}")

# ==========================================
# 3. MODEL THE PLANET WITH BATMAN
# ==========================================
# Define the physical parameters of the planet system
params = batman.TransitParams()

params.t0 = float(t0.value)                   # Time of mid-transit
params.per = float(best_period.value)              # Orbital period (days)
duration = float(duration.value)
params.rp = 0.1                        # Planet radius (in units of stellar radii)
params.a = 15.0                        # Semi-major axis (in units of stellar radii)
params.inc = 87.0                      # Orbital inclination (in degrees)
params.ecc = 0.0                       # Eccentricity (0 for circular orbit)
params.w = 90.0                        # Longitude of periastron (degrees)
params.u = [0.1, 0.3]                  # Limb darkening coefficients
params.limb_dark = "quadratic"         # Limb darkening model

# Initialize the batman model using our flattened light curve time array
time_array = np.asarray(flattened_lc.time.value, dtype=float)
m = batman.TransitModel(params, time_array)
batman_flux = m.light_curve(params)
print("\nNumber of points: ", len(flattened_lc))
print(" \nCalculating periodogram...")
# ==========================================
# 4. PLOT THE RESULTS
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Raw CSV data with the starspot trend line
axes[0].scatter(lc.time.value, lc.flux.value, color='black', s=2, label='Your CSV Data')
axes[0].plot(trend_lc.time.value, trend_lc.flux.value, color='red', lw=2, label='Starspot Trend')
axes[0].set_title("1. Raw CSV Data & Starspot Trend")
axes[0].legend()

# Plot 2: Cleaned data vs Batman Planet Model
axes[1].scatter(flattened_lc.time.value, flattened_lc.flux.value, color='blue', s=2, label='Detrended Data')
axes[1].plot(flattened_lc.time.value, batman_flux, color='orange', lw=2, label='Batman Planet Model')
axes[1].set_title("2. Cleaned Data vs Planet Model")
axes[1].legend()

# Plot 3: Zoomed-in Folded Transit Shape
folded_lc = flattened_lc.fold(period=best_period, epoch_time=t0)
# Re-calculate batman flux folded to match the plotting
folded_model_time = np.linspace(-duration, duration, 500)
m_fold = batman.TransitModel(
    params,
    folded_model_time + float(t0.value)
)
batman_flux_folded = m_fold.light_curve(params)

axes[2].scatter(folded_lc.time.value, folded_lc.flux.value, color='purple', s=4, label='Folded Data')
axes[2].plot(folded_model_time, batman_flux_folded, color='orange', lw=2, label='Batman Shape')
axes[2].set_xlim(-duration*1.5, duration*1.5) # Zoom closely into the transit dip
axes[2].set_title("3. Zoomed Folded Planet Transit Shape")
axes[2].legend()

plt.tight_layout()
plt.show()
