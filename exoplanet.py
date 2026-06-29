import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightkurve as lk
import batman
from scipy.stats import median_abs_deviation
from astropy.stats import sigma_clip
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
# ============================================================
# CONFIGURATION — CHANGE THESE ONLY
# ============================================================
USE_CSV   = False          # True = use your CSV, False = TESS download
CSV_FILE  = 'info.csv'     # your CSV filename (if USE_CSV = True)
STAR_ID   = 'TIC 261136679'  # TESS Star ID (if USE_CSV = False)
SECTOR    = 1              # which sector to download (0-10)
# ============================================================

BG     = '#0A0E1A'
PANEL  = '#0F1628'
BLUE   = '#5B7BE0'
GREEN  = '#4ECCA8'
PINK   = '#F472B6'
YELLOW = '#FBBF24'
PURPLE = '#C084FC'
GRAY   = '#aaaaaa'
RED    = '#FF6B6B'

print("="*50)
print("  EXOPLANET DETECTION SYSTEM")
print("  ISRO Bharatiya Antariksh Hackathon 2026")
print("="*50)

# ============================================================
# 1. LOAD DATA
# ============================================================
if USE_CSV:
    print(f"\nLoading CSV: {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)
    if 'QUALITY' in df.columns:
        df = df[df['QUALITY'] == 0]
    time = df['TIME'].values
    flux = df['PDCSAP_FLUX'].values
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    flux = flux / np.nanmedian(flux)
    lc = lk.LightCurve(time=time, flux=flux)
    lc = lc.remove_nans().remove_outliers(sigma=5)
    star_label = CSV_FILE
    print(f"CSV loaded! Total points: {len(lc)}")
else:
    print(f"\nDownloading TESS data for {STAR_ID} ...")
    search = lk.search_lightcurve(STAR_ID, mission="TESS")
    lc_collection = search[SECTOR].download()
    lc = lc_collection.normalize().remove_nans().remove_outliers(sigma=5)
    star_label = STAR_ID
    print(f"Downloaded! Total points: {len(lc)}")

# ============================================================
# 2. NOISE REDUCTION - 3 LAYERS
# ============================================================
print("\nReducing noise...")
# Layer 1
lc_clean = lc.remove_outliers(sigma=4)
print(f"  Layer 1 - Outlier removal: {len(lc_clean)} points")

# Layer 2
flattened_lc, trend_lc = lc_clean.flatten(
    window_length=301,
    polyorder=2,
    return_trend=True,
    break_tolerance=5
)
print(f"  Layer 2 - Starspot flattening: done")

# Layer 3
flux_vals  = np.array(flattened_lc.flux.value, dtype=float)
clipped    = sigma_clip(flux_vals, sigma=3, maxiters=5, masked=True)
clean_mask = ~np.ma.getmaskarray(clipped)
lc_final   = lk.LightCurve(
    time=flattened_lc.time[clean_mask],
    flux=flattened_lc.flux[clean_mask]
)

# Add after Layer 3
flux_smooth = savgol_filter(
    flattened_lc.flux.value,
    window_length=11,
    polyorder=2
)
lc_final = lk.LightCurve(
    time=flattened_lc.time,
    flux=flux_smooth
)
print("  Savitzky-Golay filter applied!")
print(f"  Layer 3 - Sigma clipping: {len(lc_final)} points")
print("Noise reduction complete!")
# Noise level calculation
raw_std   = np.std(lc.flux.value)
clean_std = np.std(lc_final.flux.value)
noise_reduction_pct = ((raw_std - clean_std) / raw_std * 100) if raw_std > 0 else 0
print(f"  Noise reduced by: {noise_reduction_pct:.1f}%")

flattened_lc = lc_final

# ============================================================
# 3. BLS TRANSIT DETECTION
# ============================================================
print("\nRunning BLS transit detection...")
periodogram = flattened_lc.to_periodogram(
    method='bls',
    minimum_period=1.0,
maximum_period=12,
frequency_factor=5000
)
best_period    = periodogram.period_at_max_power
t0             = periodogram.transit_time_at_max_power
duration       = periodogram.duration_at_max_power
depth          = periodogram.depth_at_max_power
duration_hours = float(duration.value) * 24

print(f"\n{'='*50}")
print(f"  TRANSIT DETECTION RESULTS")
print(f"{'='*50}")
print(f"  Period   : {best_period.value:.4f} days")
print(f"  T0       : {t0.value:.4f} BTJD")
print(f"  Duration : {duration_hours:.2f} hours")
print(f"  Depth    : {depth:.6f}")

# ============================================================
# 4. SNR CALCULATION
# ============================================================
folded      = flattened_lc.fold(period=best_period, epoch_time=t0)
half_dur    = duration.value / 2
in_transit  = np.abs(folded.time.value) < half_dur
out_transit = ~in_transit
in_flux     = folded.flux.value[in_transit]
out_flux    = folded.flux.value[out_transit]

if len(in_flux) > 0 and len(out_flux) > 0:
    transit_depth_measured = 1 - np.median(in_flux)
    rms_noise = median_abs_deviation(out_flux)
    snr = transit_depth_measured / rms_noise if rms_noise > 0 else 0
else:
    snr = 0

print(f"  SNR      : {snr:.2f}")
print(f"  RMS Noise: {rms_noise:.6f}" if len(out_flux) > 0 else "  RMS Noise: N/A")

# ============================================================
# 5. 4-CLASS CLASSIFICATION
# ============================================================
def classify_signal(depth, duration_hours, snr, period_days):
    score = {
        "Planet Transit"   : 0,
        "Eclipsing Binary" : 0,
        "Starspot"         : 0,
        "Noise"            : 0
    }
    reasons = []

    # SNR rules
    if snr < 3:
        score["Noise"] += 60
        reasons.append(f"SNR={snr:.2f} is too low (<3)")
    elif 3 <= snr < 7:
        score["Noise"] += 20
        score["Planet Transit"] += 10
        reasons.append(f"SNR={snr:.2f} is moderate (3-7)")
    else:
        score["Planet Transit"] += 40
        reasons.append(f"SNR={snr:.2f} is strong (>7)")

    # Depth rules
    if depth > 0.05:
        score["Eclipsing Binary"] += 60
        reasons.append(f"Depth={depth:.6f} very deep (>5% = binary)")
    elif 0.001 <= depth <= 0.05:
        score["Planet Transit"] += 40
        reasons.append(f"Depth={depth:.6f} is planet-like (0.1-5%)")
    else:
        score["Noise"] += 30
        reasons.append(f"Depth={depth:.6f} too shallow (<0.1%)")

    # Duration rules
    if duration_hours > 12:
        score["Starspot"] += 50
        reasons.append(f"Duration={duration_hours:.2f}h very long (starspot)")
    elif 8 < duration_hours <= 12:
        score["Starspot"] += 25
        score["Eclipsing Binary"] += 15
        reasons.append(f"Duration={duration_hours:.2f}h long (possible starspot)")
    elif 1 <= duration_hours <= 8:
        score["Planet Transit"] += 30
        reasons.append(f"Duration={duration_hours:.2f}h normal (planet-like)")
    else:
        score["Noise"] += 20
        reasons.append(f"Duration={duration_hours:.2f}h too short")

    # Period rules
    if period_days < 1:
        score["Eclipsing Binary"] += 40
        reasons.append(f"Period={period_days:.4f}d very short (binary)")
    elif 1 <= period_days <= 15:
        score["Planet Transit"] += 30
        reasons.append(f"Period={period_days:.4f}d normal (planet-like)")
    else:
        score["Starspot"] += 20
        reasons.append(f"Period={period_days:.4f}d long (starspot)")

    signal_type = max(score, key=score.get)
    total       = sum(score.values())
    confidence  = round((score[signal_type] / total * 100), 1) if total > 0 else 0
    conf_label  = "HIGH" if confidence >= 80 else "MEDIUM" if confidence >= 60 else "LOW"

    return signal_type, confidence, conf_label, score, reasons

signal_type, confidence, conf_label, scores, reasons = classify_signal(
    depth=float(depth),
    duration_hours=duration_hours,
    snr=snr,
    period_days=float(best_period.value)
)

print(f"\n{'='*50}")
print(f"  CLASSIFICATION RESULT")
print(f"{'='*50}")
print(f"  Signal Type : {signal_type}")
print(f"  Confidence  : {confidence}% ({conf_label})")
print(f"  SNR         : {snr:.2f}")
print(f"\n  WHY THIS CLASSIFICATION:")
for i, r in enumerate(reasons, 1):
    print(f"  {i}. {r}")

# ============================================================
# 6. BATMAN MODEL
# ============================================================
params           = batman.TransitParams()
params.t0        = float(t0.value)
params.per       = float(best_period.value)
params.rp        = np.sqrt(float(depth)) if depth > 0 else 0.1
params.a         = 15.0
params.inc       = 87.0
params.ecc       = 0.0
params.w         = 90.0
params.u         = [0.1, 0.3]
params.limb_dark = "quadratic"

time_array  = np.asarray(flattened_lc.time.value, dtype=float)
m           = batman.TransitModel(params, time_array)
batman_flux = m.light_curve(params)

# Planet radius in Earth radii (approximate)
rp_rs = float(params.rp)
rp_earth = rp_rs * 109.2  # solar radii to earth radii approx

# ============================================================
# 7. INTERACTIVE PLOTLY GRAPH
# ============================================================
print("\nGenerating interactive plots...")

# Signal type color
sig_colors = {
    "Planet Transit"   : GREEN,
    "Eclipsing Binary" : YELLOW,
    "Starspot"         : PINK,
    "Noise"            : BLUE
}
sig_color = sig_colors.get(signal_type, GREEN)

fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=(
        "1. Raw Light Curve & Starspot Trend (hover for details)",
        "2. Noise-Reduced Data vs BATMAN Transit Model",
        "3. Phase-Folded Transit (zoomed in)",
        "4. Signal Classification Scores"
    ),
    vertical_spacing=0.1,
    row_heights=[0.25, 0.25, 0.25, 0.25]
)

# Plot 1: Raw + Trend
fig.add_trace(go.Scatter(
    x=lc.time.value, y=lc.flux.value,
    mode='markers',
    marker=dict(size=1.5, color=BLUE, opacity=0.6),
    name='Raw Flux',
    hovertemplate=(
        '<b>RAW DATA</b><br>'
        'Time: %{x:.4f} days<br>'
        'Flux: %{y:.6f}<br>'
        f'Star: {star_label}<br>'
        '<extra></extra>'
    )
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=trend_lc.time.value, y=trend_lc.flux.value,
    mode='lines',
    line=dict(color=PINK, width=2),
    name='Starspot Trend',
    hovertemplate=(
        '<b>STARSPOT TREND</b><br>'
        'Time: %{x:.4f} days<br>'
        'Trend: %{y:.6f}<br>'
        '<extra></extra>'
    )
), row=1, col=1)

# Plot 2: Clean + BATMAN
fig.add_trace(go.Scatter(
    x=flattened_lc.time.value, y=flattened_lc.flux.value,
    mode='markers',
    marker=dict(size=1.5, color=GREEN, opacity=0.6),
    name='Cleaned Data',
    hovertemplate=(
        '<b>CLEANED DATA</b><br>'
        'Time: %{x:.4f} days<br>'
        'Flux: %{y:.6f}<br>'
        f'Noise reduced by: {noise_reduction_pct:.1f}%<br>'
        '<extra></extra>'
    )
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=flattened_lc.time.value, y=batman_flux,
    mode='lines',
    line=dict(color=YELLOW, width=2.5),
    name='BATMAN Model',
    hovertemplate=(
        '<b>BATMAN TRANSIT MODEL</b><br>'
        'Time: %{x:.4f} days<br>'
        'Model Flux: %{y:.6f}<br>'
        f'Period: {best_period.value:.4f} days<br>'
        f'Depth: {depth:.6f}<br>'
        '<extra></extra>'
    )
), row=2, col=1)

# Plot 3: Folded
fig.add_trace(go.Scatter(
    x=folded.time.value, y=folded.flux.value,
    mode='markers',
    marker=dict(size=3, color=PURPLE, opacity=0.8),
    name='Phase Folded',
    hovertemplate=(
        '<b>PHASE FOLDED TRANSIT</b><br>'
        'Phase: %{x:.4f} days<br>'
        'Flux: %{y:.6f}<br>'
        f'Period: {best_period.value:.4f} days<br>'
        f'Duration: {duration_hours:.2f} hrs<br>'
        f'Depth: {depth:.6f}<br>'
        f'SNR: {snr:.2f}<br>'
        '<extra></extra>'
    )
), row=3, col=1)

fig.add_hline(
    y=1-float(depth),
    line_dash='dash', line_color=PINK, line_width=2,
    annotation_text=f'Transit Depth = {depth:.6f}',
    annotation_font_color=PINK,
    row=3, col=1
)

fig.add_vline(
    x=0,
    line_dash='dot', line_color=YELLOW, line_width=2,
    annotation_text='Transit Centre (T0)',
    annotation_font_color=YELLOW,
    row=3, col=1
)

fig.update_xaxes(
    range=[-duration.value*3, duration.value*3],
    row=3, col=1
)

# Plot 4: Classification
categories = list(scores.keys())
values     = list(scores.values())
bar_clrs   = [GREEN, PINK, YELLOW, BLUE]

fig.add_trace(go.Bar(
    x=categories, y=values,
    marker_color=bar_clrs,
    marker_line_color='white',
    marker_line_width=1.5,
    name='Scores',
    text=[f'{v} pts' for v in values],
    textposition='outside',
    textfont=dict(color='white', size=13),
    hovertemplate=(
        '<b>%{x}</b><br>'
        'Score: %{y}<br>'
        f'Winner: {signal_type}<br>'
        f'Confidence: {confidence}% ({conf_label})<br>'
        '<extra></extra>'
    )
), row=4, col=1)

# Highlight winner bar
winner_idx = values.index(max(values))
bar_clrs[winner_idx] = sig_color

# Layout
fig.update_layout(
    title=dict(
        text=(
            f'<b>EXOPLANET DETECTION ANALYSIS</b><br>'
            f'<span style="font-size:13px; color:{sig_color}">'
            f'Star: {star_label}  |  '
            f'Signal: <b>{signal_type}</b>  |  '
            f'Confidence: <b>{confidence}% ({conf_label})</b>  |  '
            f'SNR: <b>{snr:.2f}</b>  |  '
            f'Period: <b>{best_period.value:.4f} days</b>  |  '
            f'Duration: <b>{duration_hours:.2f} hrs</b>  |  '
            f'Noise Reduced: <b>{noise_reduction_pct:.1f}%</b>'
            f'</span>'
        ),
        font=dict(size=16, color='white'),
        x=0.5
    ),
    height=1500,
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        bgcolor=PANEL,
        bordercolor=BLUE,
        borderwidth=1,
        font=dict(color='white')
    ),
    hovermode='x unified'
)

fig.update_xaxes(
    gridcolor='rgba(255,255,255,0.1)',
    zerolinecolor='rgba(255,255,255,0.2)',
    color=GRAY
)
fig.update_yaxes(
    gridcolor='rgba(255,255,255,0.1)',
    zerolinecolor='rgba(255,255,255,0.2)',
    color=GRAY
)

# Save HTML
html_file = 'D:\\isro\\exoplanet_interactive.html'
fig.write_html(
    html_file,
    config={
        'scrollZoom': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
    }
)
print(f"Interactive HTML saved!")

# Save PNG
png_file = 'D:\\isro\\exoplanet_results.png'
try:
    fig.write_image(png_file, width=1600, height=1500)
    print(f"PNG saved!")
except:
    print("PNG save skipped (kaleido issue)")

fig.show()

# ============================================================
# 8. FINAL SUMMARY REPORT
# ============================================================
print(f"\n{'='*50}")
print(f"  FINAL DETECTION REPORT")
print(f"{'='*50}")
print(f"  Star ID       : {star_label}")
print(f"  Signal Type   : {signal_type}")
print(f"  Confidence    : {confidence}% ({conf_label})")
print(f"  SNR           : {snr:.2f}")
print(f"{'='*50}")
print(f"  ORBITAL PARAMETERS")
print(f"{'='*50}")
print(f"  Period        : {best_period.value:.4f} days")
print(f"  Duration      : {duration_hours:.2f} hours")
print(f"  Depth         : {depth:.6f}")
print(f"  Rp/Rs         : {rp_rs:.4f}")
print(f"  Planet Radius : ~{rp_earth:.1f} Earth radii")
print(f"{'='*50}")
print(f"  NOISE ANALYSIS")
print(f"{'='*50}")
print(f"  Raw Noise     : {raw_std:.6f}")
print(f"  Clean Noise   : {clean_std:.6f}")
print(f"  Noise Reduced : {noise_reduction_pct:.1f}%")
print(f"{'='*50}")
print(f"  CLASSIFICATION REASONS")
print(f"{'='*50}")
for i, r in enumerate(reasons, 1):
    print(f"  {i}. {r}")
print(f"{'='*50}")

# ============================================================
# 9. SAVE TO CSV DATABASE
# ============================================================
db_file = 'D:\\isro\\results_database.csv'

new_result = {
    'DateTime'          : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'StarID'            : star_label,
    'SignalType'        : signal_type,
    'Confidence_pct'    : confidence,
    'ConfLevel'         : conf_label,
    'SNR' : round(float(snr), 2),
    'Period_days'       : round(float(best_period.value), 4),
    'Duration_hrs'      : round(duration_hours, 2),
    'Depth'             : round(float(depth), 6),
    'Rp_Rs'             : round(rp_rs, 4),
    'Planet_Radius_Earth': round(rp_earth, 1),
    'RawNoise'          : round(raw_std, 6),
    'CleanNoise'        : round(clean_std, 6),
    'NoiseReduced_pct'  : round(noise_reduction_pct, 1),
    'PlotHTML'          : html_file,
    'PlotPNG'           : png_file
}

if os.path.exists(db_file):
    db      = pd.read_csv(db_file)
    new_row = pd.DataFrame([new_result])
    db      = pd.concat([db, new_row], ignore_index=True)
else:
    db = pd.DataFrame([new_result])

db.to_csv(db_file, index=False)

print(f"\n{'='*50}")
print(f"  DATABASE SAVED!")
print(f"{'='*50}")
print(f"  File          : results_database.csv")
print(f"  Total Records : {len(db)}")
print(f"  Latest Entry  : {signal_type} | {confidence}% | SNR={snr:.2f}")
print(f"{'='*50}")
print(f"\nDONE! Open this file in browser:")
print(f"  {html_file}")