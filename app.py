import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightkurve as lk
import batman
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# ── Optional deps for extra features (fail soft if missing) ────
try:
    from astroquery.mast import Catalogs
    ASTROQUERY_OK = True
except Exception:
    ASTROQUERY_OK = False

try:
    import anthropic
    ANTHROPIC_SDK_OK = True
except Exception:
    ANTHROPIC_SDK_OK = False

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    try:
        ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        ANTHROPIC_API_KEY = ""

# ── Try loading real CNN model ─────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "exoplanet_classifier.h5")
META_PATH = os.path.join(BASE_DIR, "model_meta.csv")
CNN_LOADED = False
cnn_model  = None
CLASS_NAMES = ['Eclipsing Binary', 'Noise', 'Planet Transit', 'Starspot']
N_POINTS   = 201

CNN_LOADED = False
try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        cnn_model = tf.keras.models.load_model(MODEL_PATH)
        CNN_LOADED = True
        if os.path.exists(META_PATH):
            meta = pd.read_csv(META_PATH)
            import ast
            CLASS_NAMES = ast.literal_eval(meta['class_names'].iloc[0])
except Exception as e:
    st.warning(f"CNN not loaded: {e} — using rule-based fallback")
# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BG     = '#0A0E1A'
PANEL  = '#0F1628'
BLUE   = '#5B7BE0'
GREEN  = '#4ECCA8'
PINK   = '#F472B6'
YELLOW = '#FBBF24'
PURPLE = '#C084FC'
GRAY   = '#aaaaaa'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{{font-family:'Space Grotesk',sans-serif;background:{BG};color:white}}
#MainMenu,footer,header{{visibility:hidden}}
.block-container{{padding-top:1rem;padding-bottom:2rem}}
.navbar{{display:flex;align-items:center;justify-content:space-between;
         background:{PANEL};border-bottom:1px solid {BLUE}44;
         padding:0 2.5rem;height:58px;margin-bottom:2rem;border-radius:12px}}
.nav-brand{{font-family:'JetBrains Mono',monospace;font-size:1.05rem;
            font-weight:700;color:{GREEN};letter-spacing:1px}}
.glass-card{{background:{PANEL};border:1px solid {BLUE}33;
             border-radius:14px;padding:1.8rem 2rem;margin-bottom:1.2rem}}
.input-label{{font-size:.72rem;font-weight:600;text-transform:uppercase;
              letter-spacing:2px;color:{GRAY};margin-bottom:.5rem}}
.input-desc{{font-size:.85rem;color:{GRAY};margin-bottom:1rem;line-height:1.5}}
.metric-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1rem 0}}
.metric-box{{background:{BG};border:1px solid {BLUE}33;border-radius:10px;
             padding:1rem;text-align:center}}
.metric-label{{font-size:.72rem;color:{GRAY};text-transform:uppercase;
               letter-spacing:1px;margin-bottom:4px}}
.metric-value{{font-family:'JetBrains Mono',monospace;font-size:1.3rem;
               font-weight:700;color:white}}
.result-badge{{display:inline-block;padding:6px 18px;border-radius:30px;font-weight:700;font-size:1rem}}
.reason-item{{background:{BG};border-left:3px solid {BLUE};border-radius:0 8px 8px 0;
              padding:8px 14px;margin:6px 0;font-size:.85rem;color:#ccc}}
.step-item{{display:flex;align-items:center;gap:12px;padding:10px 0;
            border-bottom:1px solid {BLUE}22;font-size:.9rem}}
.step-num{{width:28px;height:28px;border-radius:50%;background:{BLUE}22;
           border:1px solid {BLUE};display:flex;align-items:center;
           justify-content:center;font-size:.75rem;font-weight:700;
           color:{BLUE};flex-shrink:0}}
.arch-node{{background:{PANEL};border:1.5px solid {BLUE}44;border-radius:12px;
            padding:14px 18px;margin:5px 0;display:flex;align-items:center;gap:14px}}
.arch-arrow{{text-align:center;color:{BLUE}66;font-size:1.1rem;margin:2px 0}}
.arch-badge{{margin-left:auto;font-size:.65rem;font-weight:700;padding:3px 10px;
             border-radius:20px;letter-spacing:.5px}}
.section-title{{font-size:.7rem;font-weight:600;text-transform:uppercase;
                letter-spacing:2px;color:{GRAY};margin-bottom:.8rem;margin-top:1.2rem}}
.stTextInput input{{background:{BG}!important;border:1px solid {BLUE}66!important;
                    border-radius:10px!important;color:white!important;
                    font-family:'JetBrains Mono',monospace!important;
                    font-size:1rem!important;padding:.6rem 1rem!important}}
.stButton>button{{background:linear-gradient(135deg,{BLUE},{PURPLE})!important;
                  color:white!important;border:none!important;border-radius:10px!important;
                  font-weight:600!important;padding:.65rem 2rem!important;
                  width:100%!important;font-size:1rem!important}}
.success-box{{background:{GREEN}11;border:1px solid {GREEN}44;border-radius:10px;
              padding:1rem 1.4rem;margin:1rem 0;color:{GREEN};font-weight:500}}
.wait-box{{background:{YELLOW}11;border:1px solid {YELLOW}44;border-radius:10px;
           padding:1rem 1.4rem;color:{YELLOW};font-size:.9rem;margin:1rem 0}}
.model-badge{{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;
              border-radius:20px;font-size:.78rem;font-weight:600;margin-bottom:1rem}}
</style>
""", unsafe_allow_html=True)

if 'page'    not in st.session_state: st.session_state.page    = 'detect'
if 'results' not in st.session_state: st.session_state.results = None

# Navbar
model_status = (f'<span style="color:{GREEN}">● CNN Active</span>'
                if CNN_LOADED else
                f'<span style="color:{YELLOW}">● Rule-Based (train CNN first)</span>')
st.markdown(f"""
<div class="navbar">
  <div class="nav-brand">⬡ EXOPLANET · AI</div>
  <div style="font-size:.8rem;color:{GRAY}">
    AI-Powered Transit Detection &nbsp;|&nbsp; {model_status}
  </div>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_graph, tab_db, tab_arch, tab_3d, tab_ai, tab_extra = st.tabs([
    "🔭  Detect", "📊  Graph", "🗄️  Database", "🏗️  Architecture",
    "🌍  3D Orbit", "🤖  AI Analysis", "🔬  Deep Analysis"
])

# ── CLASSIFICATION ─────────────────────────────────────────────
def classify_cnn(folded_flux, snr, depth, period_days, dur_hours):
    """Real CNN classification"""
    # Normalise folded flux to N_POINTS
    from scipy.interpolate import interp1d
    x_old = np.linspace(0, 1, len(folded_flux))
    x_new = np.linspace(0, 1, N_POINTS)
    f_interp = interp1d(x_old, folded_flux, kind='linear', fill_value='extrapolate')
    signal = f_interp(x_new).astype(np.float32)

    # Normalise to [0,1]
    sig_min, sig_max = signal.min(), signal.max()
    signal = (signal - sig_min) / (sig_max - sig_min + 1e-8)
    signal = signal.reshape(1, N_POINTS, 1)

    probs = cnn_model.predict(signal, verbose=0)[0]
    idx   = int(np.argmax(probs))
    sig   = CLASS_NAMES[idx]
    conf  = round(float(probs[idx]) * 100, 1)
    conf_lbl = 'HIGH' if conf >= 80 else 'MEDIUM' if conf >= 60 else 'LOW'

    score = {CLASS_NAMES[i]: int(round(probs[i]*100)) for i in range(4)}
    reasons = [
        f'CNN probability: {sig} = {conf:.1f}%',
        f'SNR = {snr:.2f}  |  Depth = {depth:.6f}',
        f'Period = {period_days:.4f} d  |  Duration = {dur_hours:.2f} h',
        f'Model: Conv1D x3 → Dense → Softmax (4 classes)',
    ]
    return sig, conf, conf_lbl, score, reasons


def classify_rules(depth, dur_hours, snr, period_days):
    """Rule-based fallback (clearly labeled as such)"""
    score   = {'Planet Transit':0,'Eclipsing Binary':0,'Starspot':0,'Noise':0}
    reasons = ['[RULE-BASED — train CNN for ML classification]']

    if snr < 3:
        score['Noise'] += 60; reasons.append(f'SNR {snr:.2f} < 3 — weak signal')
    elif snr < 7:
        score['Noise'] += 20; score['Planet Transit'] += 10
        reasons.append(f'SNR {snr:.2f} moderate')
    else:
        score['Planet Transit'] += 40; reasons.append(f'SNR {snr:.2f} strong')

    if depth > 0.05:
        score['Eclipsing Binary'] += 60; reasons.append(f'Depth {depth:.6f} very deep (>5%)')
    elif depth >= 0.001:
        score['Planet Transit'] += 40; reasons.append(f'Depth {depth:.6f} planet-like')
    else:
        score['Noise'] += 30; reasons.append(f'Depth {depth:.6f} too shallow')

    if dur_hours > 12:
        score['Starspot'] += 50; reasons.append(f'Duration {dur_hours:.2f}h → starspot')
    elif dur_hours >= 1:
        score['Planet Transit'] += 30; reasons.append(f'Duration {dur_hours:.2f}h normal')
    else:
        score['Noise'] += 20; reasons.append(f'Duration {dur_hours:.2f}h too short')

    if period_days < 1:
        score['Eclipsing Binary'] += 40; reasons.append(f'Period {period_days:.4f}d short → binary')
    elif period_days <= 15:
        score['Planet Transit'] += 30; reasons.append(f'Period {period_days:.4f}d normal')
    else:
        score['Starspot'] += 20; reasons.append(f'Period {period_days:.4f}d long → starspot')

    sig      = max(score, key=score.get)
    tot      = sum(score.values())
    conf     = round(score[sig]/tot*100, 1) if tot > 0 else 0
    conf_lbl = 'HIGH' if conf >= 80 else 'MEDIUM' if conf >= 60 else 'LOW'
    return sig, conf, conf_lbl, score, reasons


# ═════════════════════════════════════════════════════════════
# FEATURE — Real BATMAN Parameter Fitting
# ═════════════════════════════════════════════════════════════
def fit_batman_model(flat_time, flat_flux, t0_val, period_val, depth_val):
    """Fits BATMAN transit parameters to real light curve data using scipy minimize."""
    rp_init = float(np.sqrt(depth_val)) if depth_val > 0 else 0.1
    a_init  = (period_val ** (2/3)) * (1.0 ** (1/3)) * 215.0
    a_init  = max(3.0, min(a_init / 10, 50.0))

    x0 = [rp_init, a_init, 87.0, 0.3, 0.1]
    bounds = [
        (0.001, 0.5), (2.0, 100.0), (70.0, 90.0), (0.0, 1.0), (0.0, 1.0),
    ]

    ta = np.asarray(flat_time, dtype=float)
    fl = np.asarray(flat_flux, dtype=float)

    def residuals(x):
        rp, a, inc, u1, u2 = x
        try:
            pm = batman.TransitParams()
            pm.t0, pm.per, pm.rp, pm.a, pm.inc = t0_val, period_val, rp, a, inc
            pm.ecc, pm.w = 0.0, 90.0
            pm.u, pm.limb_dark = [u1, u2], 'quadratic'
            model_flux = batman.TransitModel(pm, ta).light_curve(pm)
            return float(np.sum((fl - model_flux) ** 2))
        except Exception:
            return 1e10

    result = minimize(residuals, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12})

    rp_fit, a_fit, inc_fit, u1_fit, u2_fit = result.x
    pm = batman.TransitParams()
    pm.t0, pm.per, pm.rp, pm.a, pm.inc = t0_val, period_val, rp_fit, a_fit, inc_fit
    pm.ecc, pm.w = 0.0, 90.0
    pm.u, pm.limb_dark = [u1_fit, u2_fit], 'quadratic'

    bf = batman.TransitModel(pm, ta).light_curve(pm)

    ss_res = np.sum((fl - bf) ** 2)
    ss_tot = np.sum((fl - np.mean(fl)) ** 2)
    r2     = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    fit_params = {
        'rp': round(rp_fit, 5), 'a': round(a_fit, 3), 'inc': round(inc_fit, 3),
        'u1': round(u1_fit, 4), 'u2': round(u2_fit, 4),
        'r_squared': round(r2, 4), 'converged': result.success,
    }
    return pm, bf, fit_params


# ═════════════════════════════════════════════════════════════
# FEATURE — Star Properties from NASA TIC Catalog
# ═════════════════════════════════════════════════════════════
def fetch_star_properties(star_id):
    defaults = {'radius_solar': 1.0, 'temp_k': 5778, 'mass_solar': 1.0,
                'logg': 4.44, 'found': False}
    if not ASTROQUERY_OK:
        return defaults
    try:
        tic_id = star_id.strip().upper().replace('TIC', '').strip()
        result = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id))
        if len(result) == 0:
            return defaults
        row = result[0]
        return {
            'radius_solar': float(row['rad']) if row['rad'] else 1.0,
            'temp_k'      : float(row['Teff']) if row['Teff'] else 5778,
            'mass_solar'  : float(row['mass']) if row['mass'] else 1.0,
            'logg'        : float(row['logg']) if row['logg'] else 4.44,
            'found'       : True,
        }
    except Exception:
        return defaults


# ═════════════════════════════════════════════════════════════
# FEATURE — Adaptive BLS Period Range
# ═════════════════════════════════════════════════════════════
def get_adaptive_period_range(flat_lc):
    baseline   = float(flat_lc.time.value[-1] - flat_lc.time.value[0])
    min_period = 0.5
    max_period = min(baseline / 3.0, 100.0)
    max_period = max(max_period, 2.0)
    return min_period, max_period


# ═════════════════════════════════════════════════════════════
# FEATURE — Secondary Eclipse False-Positive Filter
# ═════════════════════════════════════════════════════════════
def check_secondary_eclipse(folded_lc, primary_depth, duration_val):
    try:
        phase = folded_lc.time.value
        flux  = folded_lc.flux.value
        hd    = duration_val / 2

        sec_mask = (np.abs(phase - 0.5) < hd) | (np.abs(phase + 0.5) < hd)
        out_mask = (np.abs(phase) > hd) & ~sec_mask

        if sec_mask.sum() < 3 or out_mask.sum() < 3:
            return {'has_secondary': False, 'secondary_depth': 0.0,
                    'depth_ratio': 0.0, 'verdict': 'Insufficient data'}

        sec_flux  = flux[sec_mask]
        out_flux  = flux[out_mask]
        sec_depth = float(np.median(out_flux) - np.median(sec_flux))
        ratio     = sec_depth / primary_depth if primary_depth > 0 else 0.0
        has_sec   = ratio > 0.1

        if ratio > 0.5:
            verdict = '🚨 Strong secondary eclipse — likely Eclipsing Binary'
        elif ratio > 0.1:
            verdict = '⚠️ Weak secondary eclipse — possible false positive'
        else:
            verdict = '✅ No secondary eclipse — consistent with planet'

        return {'has_secondary': has_sec, 'secondary_depth': round(sec_depth, 6),
                'depth_ratio': round(ratio, 4), 'verdict': verdict}
    except Exception:
        return {'has_secondary': False, 'secondary_depth': 0.0,
                'depth_ratio': 0.0, 'verdict': 'Check failed'}


# ═════════════════════════════════════════════════════════════
# FEATURE — Multi-Planet Detection
# ═════════════════════════════════════════════════════════════
def detect_second_planet(flat_lc, primary_period, primary_t0,
                          primary_duration, min_per, max_per):
    try:
        t  = flat_lc.time.value
        fl = flat_lc.flux.value

        phase1  = ((t - primary_t0) % primary_period) / primary_period
        phase1  = np.where(phase1 > 0.5, phase1 - 1.0, phase1)
        hd      = (primary_duration / primary_period) / 2 * 1.5
        p1_mask = np.abs(phase1) > hd

        if p1_mask.sum() < 50:
            return {'found': False}

        lc_masked = lk.LightCurve(time=flat_lc.time[p1_mask], flux=flat_lc.flux[p1_mask])

        pg2 = lc_masked.to_periodogram(
            method='bls', minimum_period=min_per, maximum_period=max_per,
            frequency_factor=3000
        )

        p2     = float(pg2.period_at_max_power.value)
        dep2   = float(pg2.depth_at_max_power)
        dur2   = float(pg2.duration_at_max_power.value)
        t0_2   = float(pg2.transit_time_at_max_power.value)
        dur2_h = dur2 * 24

        fo2 = lc_masked.fold(period=pg2.period_at_max_power,
                              epoch_time=pg2.transit_time_at_max_power)
        hd2 = dur2 / 2
        in2 = np.abs(fo2.time.value) < hd2
        if in2.sum() > 2 and (~in2).sum() > 2:
            rms2 = float(median_abs_deviation(fo2.flux.value[~in2]))
            snr2 = float((1 - np.median(fo2.flux.value[in2])) / rms2) if rms2 > 0 else 0
        else:
            snr2 = 0.0

        period_ratio = p2 / primary_period
        too_similar  = 0.95 < period_ratio < 1.05 or 0.45 < period_ratio < 0.55

        if snr2 >= 3.0 and dep2 > 0.0001 and not too_similar:
            return {
                'found': True, 'period': round(p2, 4), 'depth': round(dep2, 6),
                'duration_hrs': round(dur2_h, 2), 'snr': round(snr2, 2),
                't0': round(t0_2, 4), 'rp_rs': round(float(np.sqrt(dep2)), 4),
                'rp_earth': round(float(np.sqrt(dep2)) * 109.2, 1),
            }
        return {'found': False}
    except Exception:
        return {'found': False}


# ═════════════════════════════════════════════════════════════
# FEATURE — Next Transit Alert
# ═════════════════════════════════════════════════════════════
def compute_next_transits(t0_bjd, period_days, n=5):
    BJD_OFFSET = 2457000.0
    today_bjd  = (datetime.utcnow() - datetime(2014, 12, 8)).days + BJD_OFFSET

    cycles_elapsed = np.ceil((today_bjd - t0_bjd) / period_days)
    next_t0_bjd    = t0_bjd + cycles_elapsed * period_days

    transits = []
    for i in range(n):
        bjd_i     = next_t0_bjd + i * period_days
        days_away = bjd_i - today_bjd
        utc_date  = datetime.utcnow() + timedelta(days=days_away)
        transits.append({
            'transit_num'  : int(cycles_elapsed) + i + 1,
            'bjd'          : round(bjd_i, 4),
            'utc_date'     : utc_date.strftime('%Y-%m-%d %H:%M UTC'),
            'days_from_now': round(days_away, 1),
        })
    return transits


# ═════════════════════════════════════════════════════════════
# FEATURE — Period vs Radius Database Chart
# ═════════════════════════════════════════════════════════════
def make_database_chart(db):
    color_map = {'Planet Transit': GREEN, 'Eclipsing Binary': YELLOW,
                 'Starspot': PINK, 'Noise': GRAY}
    fig = go.Figure()
    for sig_type, grp in db.groupby('SignalType'):
        clr = color_map.get(sig_type, BLUE)
        fig.add_trace(go.Scatter(
            x=grp['Period_days'], y=grp['Rp_Earth'], mode='markers+text',
            name=sig_type, text=grp['StarID'], textposition='top center',
            textfont=dict(size=8, color=clr),
            marker=dict(size=grp['SNR'].clip(3, 20) * 1.5, color=clr, opacity=0.85,
                        line=dict(color='white', width=0.5), symbol='circle'),
            hovertemplate=('<b>%{text}</b><br>Period: %{x:.3f} d<br>'
                           f'Radius: %{{y:.1f}} R⊕<br>Type: {sig_type}<extra></extra>')
        ))
    fig.add_hrect(y0=0.8, y1=2.0, fillcolor=GREEN, opacity=0.04, line_width=0,
                  annotation_text='Earth-size range', annotation_font_color=GREEN,
                  annotation_position='top left')
    fig.update_layout(
        title=dict(text='All Detections — Period vs Planet Radius',
                   font=dict(color='white', size=14), x=0.5),
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(title='Orbital Period (days)', color=GRAY,
                   gridcolor='#ffffff11', type='log', showgrid=True),
        yaxis=dict(title='Planet Radius (R⊕)', color=GRAY, gridcolor='#ffffff11'),
        legend=dict(font=dict(color='white'), bgcolor=PANEL,
                    bordercolor=BLUE, borderwidth=1),
        height=460, margin=dict(l=60, r=20, t=50, b=60), hovermode='closest'
    )
    return fig


# ═════════════════════════════════════════════════════════════
# FEATURE — 3D Interactive Planet Orbit Visualization
# ═════════════════════════════════════════════════════════════
def make_3d_orbit(period_days, rp_rs, inclination_deg=87.0,
                   star_radius=1.0, signal_type='Planet Transit'):
    inc_rad  = np.radians(inclination_deg)
    a_au     = period_days ** (2/3) * (1.0 ** (1/3))
    a_plot   = a_au * 10
    r_star   = star_radius * 1.0
    r_planet = rp_rs * r_star * 8

    planet_clr = {
        'Planet Transit': '#4ECCA8', 'Eclipsing Binary': '#FBBF24',
        'Starspot': '#F472B6', 'Noise': '#aaaaaa',
    }.get(signal_type, '#4ECCA8')

    theta   = np.linspace(0, 2 * np.pi, 300)
    orbit_x = a_plot * np.cos(theta)
    orbit_y = a_plot * np.sin(theta) * np.cos(inc_rad)
    orbit_z = a_plot * np.sin(theta) * np.sin(inc_rad)

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    sx = r_star * np.cos(u) * np.sin(v)
    sy = r_star * np.sin(u) * np.sin(v)
    sz = r_star * np.cos(v)

    px_center, py_center, pz_center = a_plot, 0.0, 0.0
    px = r_planet * np.cos(u) * np.sin(v) + px_center
    py = r_planet * np.sin(u) * np.sin(v) + py_center
    pz = r_planet * np.cos(v) + pz_center

    n_frames = 36
    frames = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        fx = a_plot * np.cos(angle)
        fy = a_plot * np.sin(angle) * np.cos(inc_rad)
        fz = a_plot * np.sin(angle) * np.sin(inc_rad)
        fpx = r_planet * np.cos(u) * np.sin(v) + fx
        fpy = r_planet * np.sin(u) * np.sin(v) + fy
        fpz = r_planet * np.cos(v) + fz
        frames.append(go.Frame(
            data=[go.Surface(x=fpx, y=fpy, z=fpz,
                              colorscale=[[0, planet_clr], [1, '#ffffff']],
                              showscale=False, opacity=0.95, name='Planet')],
            traces=[3], name=str(i)
        ))

    fig = go.Figure(
        data=[
            go.Scatter3d(x=orbit_x, y=orbit_y, z=orbit_z, mode='lines',
                         line=dict(color=BLUE, width=2, dash='dot'),
                         name='Orbital Path', hoverinfo='skip'),
            go.Surface(x=sx, y=sy, z=sz,
                      colorscale=[[0, '#FFA500'], [0.5, '#FFD700'], [1, '#FFFFFF']],
                      showscale=False, opacity=1.0, name='Host Star',
                      hovertemplate='Host Star<extra></extra>'),
            go.Scatter3d(x=[-r_star*1.2, r_star*1.2], y=[0,0], z=[0,0], mode='lines',
                        line=dict(color=GREEN, width=6), name='Transit Zone', hoverinfo='skip'),
            go.Surface(x=px, y=py, z=pz, colorscale=[[0, planet_clr], [1, '#ffffff']],
                      showscale=False, opacity=0.95, name='Planet',
                      hovertemplate=f'Planet<br>Rp/Rs: {rp_rs:.4f}<extra></extra>'),
            go.Surface(x=sx*1.3, y=sy*1.3, z=sz*1.3,
                      colorscale=[[0, '#FF8C00'], [1, '#FF8C0000']],
                      showscale=False, opacity=0.08, name='Star Glow', hoverinfo='skip'),
        ],
        frames=frames
    )

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[r_star*1.5], mode='text', text=['⭐ Host Star'],
        textfont=dict(color=YELLOW, size=12), hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[a_plot], y=[0], z=[r_planet*2], mode='text',
        text=[f'🪐 Planet<br>{rp_rs:.4f} Rp/Rs'],
        textfont=dict(color=planet_clr, size=11), hoverinfo='skip', showlegend=False
    ))

    fig.update_layout(
        title=dict(
            text=f'3D Orbital View  |  Period: {period_days:.4f} d  |  '
                 f'Inclination: {inclination_deg:.1f}°  |  Signal: {signal_type}',
            font=dict(color='white', size=14), x=0.5
        ),
        paper_bgcolor=BG,
        scene=dict(
            bgcolor=BG,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)), aspectmode='cube'
        ),
        legend=dict(font=dict(color='white', size=11), bgcolor=PANEL,
                    bordercolor=BLUE, borderwidth=1, x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=50, b=0), height=620,
        updatemenus=[dict(
            type='buttons', showactive=False, y=1.05, x=0.05, xanchor='left',
            buttons=[
                dict(label='▶  Animate Orbit', method='animate',
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                       fromcurrent=True, transition=dict(duration=0))]),
                dict(label='⏸  Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode='immediate', transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], dict(mode='immediate',
                                             frame=dict(duration=80, redraw=True),
                                             transition=dict(duration=0))],
                       label=f'{int(360*i/n_frames)}°', method='animate')
                  for i, f in enumerate(frames)],
            transition=dict(duration=0), x=0.05, y=0,
            currentvalue=dict(font=dict(color=GRAY, size=11), prefix='Orbit position: ',
                              visible=True, xanchor='left'),
            len=0.9, bgcolor=PANEL, bordercolor=BLUE, tickcolor=GRAY,
            font=dict(color=GRAY, size=9)
        )]
    )
    return fig


# ═════════════════════════════════════════════════════════════
# AI FEATURES — Powered by Anthropic Claude API
# ═════════════════════════════════════════════════════════════
def build_planet_context(r):
    return f"""
    Detected Exoplanet Candidate Data:
    - Star: {r.get('star_label', 'Unknown')}
    - Signal Type: {r.get('signal_type', 'Unknown')}
    - Classification Confidence: {r.get('confidence', 0)}% ({r.get('conf_lbl', '')})
    - Classification Method: {r.get('method', 'Unknown')}
    - Orbital Period: {float(r['best_period'].value):.4f} days
    - Transit Duration: {r.get('dur_hours', 0):.2f} hours
    - Transit Depth: {r.get('depth', 0):.6f} (fractional flux drop)
    - Signal-to-Noise Ratio (SNR): {r.get('snr', 0):.2f}
    - Planet/Star Radius Ratio (Rp/Rs): {r.get('rp_rs', 0):.4f}
    - Planet Radius: ~{r.get('rp_earth', 0):.1f} Earth radii
    - Noise Reduction Achieved: {r.get('noise_pct', 0):.1f}%
    - Data Source: NASA TESS Mission
    """


def ai_chatbot_response(user_question, planet_context, chat_history):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_prompt = f"""You are ExoAI, an expert astrophysicist and exoplanet scientist
    embedded in an exoplanet detection app. You help users understand their detected
    exoplanet candidates in simple, exciting, and scientifically accurate terms.

    Always refer to the planet data provided. Be enthusiastic about space science.
    If a planet has high SNR and correct depth/duration, express excitement.
    If it looks like noise or a false positive, explain why gently.
    Keep responses concise (3-5 sentences) unless asked for detail.
    Use emojis occasionally to make it engaging.

    Current Detection Results:
    {planet_context}
    """
    messages = chat_history + [{"role": "user", "content": user_question}]
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1000,
        system=system_prompt, messages=messages
    )
    return response.content[0].text


def generate_ai_report(planet_context, star_id):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""You are a professional astrophysicist writing a scientific
    detection report for a peer-reviewed journal. Based on the following
    exoplanet detection data, write a structured report.

    {planet_context}

    Write the report with these sections:
    1. Abstract (2-3 sentences)
    2. Detection Summary (key parameters explained)
    3. Physical Characteristics (what we know about this planet)
    4. Comparison to Known Exoplanets (similar known planets)
    5. Habitability Assessment (could it support life?)
    6. Confidence & Limitations (what could be wrong)
    7. Recommended Follow-up Observations

    Use scientific language but keep it understandable.
    Format with markdown headers (##).
    Be specific with numbers from the data provided.
    Total length: 400-600 words.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def analyze_habitability(planet_context):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""You are an astrobiologist analyzing an exoplanet candidate for habitability.

    {planet_context}

    Analyze this planet for habitability. Respond ONLY with a JSON object, no extra text:
    {{
        "score": <integer 0-100, habitability score>,
        "verdict": "<one of: Potentially Habitable / Unlikely Habitable / Uninhabitable / Insufficient Data>",
        "zone": "<one of: Habitable Zone / Too Hot / Too Cold / Unknown>",
        "factors": [
            "<factor 1: e.g. 'Period suggests possible temperate orbit'>",
            "<factor 2>",
            "<factor 3>",
            "<factor 4>"
        ],
        "explanation": "<2-3 sentence plain-English explanation of the score>",
        "similar_planet": "<name of a known exoplanet it resembles, e.g. Kepler-452b>",
        "follow_up": "<one key observation that would confirm or deny habitability>"
    }}

    Base score on: orbital period (habitable zone), planet size, transit depth, SNR quality.
    Assume solar-type host star unless data suggests otherwise.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── PIPELINE ──────────────────────────────────────────────────
def run_pipeline(use_csv, csv_df, star_id, sector):
    if use_csv:
        df = csv_df.copy()

        required = ['TIME', 'PDCSAP_FLUX']
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if 'QUALITY' in df.columns:
            df = df[df['QUALITY'] == 0]

        t = df['TIME'].values
        fl = df['PDCSAP_FLUX'].values

        mask = np.isfinite(t) & np.isfinite(fl)
        t, fl = t[mask], fl[mask]

        fl = fl / np.nanmedian(fl)
        lc = lk.LightCurve(time=t, flux=fl)
        lc = lc.remove_nans().remove_outliers(sigma=5)
        star_label = 'Uploaded CSV'

    else:
        search = lk.search_lightcurve(star_id, mission="TESS")

        if sector >= len(search):
            raise ValueError(
                f"This star only has {len(search)} available sectors."
            )

        lc_col = search[sector].download()
        lc = lc_col.normalize().remove_nans().remove_outliers(sigma=5)
        star_label = star_id

        raw_std = float(np.std(lc.flux.value))

        # Noise reduction
    raw_std = float(np.std(lc.flux.value))

    lc_clean = lc.remove_outliers(sigma=4)
    flat, trend_lc = lc_clean.flatten(
        window_length=301,
        polyorder=2,
        return_trend=True,
        break_tolerance=5
    )

    fv = np.array(flat.flux.value, dtype=float)
    fsg = savgol_filter(fv, window_length=11, polyorder=2)
    clp = sigma_clip(fsg, sigma=3, maxiters=5, masked=True)
    mk = ~np.ma.getmaskarray(clp)

    lc_final = lk.LightCurve(
        time=flat.time[mk],
        flux=flat.flux[mk]
    )

    clean_std = float(np.std(lc_final.flux.value))
    noise_pct = (raw_std - clean_std) / raw_std * 100 if raw_std > 0 else 0
    flat = lc_final

    # BLS — Adaptive period range
    min_per, max_per = get_adaptive_period_range(flat)
    pg = flat.to_periodogram(
        method='bls',
        minimum_period=min_per,
        maximum_period=max_per,
        frequency_factor=5000
    )
    bp  = pg.period_at_max_power
    t0  = pg.transit_time_at_max_power
    dur = pg.duration_at_max_power
    dep = float(pg.depth_at_max_power)
    dur_h = float(dur.value) * 24

    # SNR
    fo   = flat.fold(period=bp, epoch_time=t0)
    hd   = dur.value / 2
    in_t = np.abs(fo.time.value) < hd
    inf  = fo.flux.value[in_t]
    outf = fo.flux.value[~in_t]
    if len(inf) > 0 and len(outf) > 0:
        rms = float(median_abs_deviation(outf))
        snr = float((1-np.median(inf))/rms) if rms > 0 else 0.0
    else:
        snr, rms = 0.0, 0.0

    # Star properties — NASA TIC Catalog
    if not use_csv:
        star_props = fetch_star_properties(star_id)
    else:
        star_props = {'radius_solar': 1.0, 'temp_k': 5778, 'mass_solar': 1.0,
                       'logg': 4.44, 'found': False}

    # Secondary eclipse check (false-positive filter)
    sec_eclipse = check_secondary_eclipse(fo, dep, float(dur.value))

    # Multi-planet detection (search residual light curve)
    planet2 = detect_second_planet(
        flat, float(bp.value), float(t0.value), float(dur.value), min_per, max_per
    )

    # Next transit alerts
    next_transits = compute_next_transits(float(t0.value), float(bp.value), n=5)

    # Classification — CNN if available, else rules
    if CNN_LOADED:
        sig, conf, conf_lbl, score, reasons = classify_cnn(
            fo.flux.value, snr, dep,
            float(bp.value), dur_h
        )
        method = 'CNN'
    else:
        sig, conf, conf_lbl, score, reasons = classify_rules(
            dep, dur_h, snr, float(bp.value)
        )
        method = 'Rule-Based'

    # BATMAN — real fitted model (scipy optimize, not hardcoded)
    ta = np.asarray(flat.time.value, dtype=float)
    fl_arr = np.asarray(flat.flux.value, dtype=float)
    pm, bf, fit_params = fit_batman_model(
        flat_time=ta, flat_flux=fl_arr,
        t0_val=float(t0.value), period_val=float(bp.value), depth_val=dep,
    )

    return dict(
        star_label=star_label, lc=lc,
        trend_lc=trend_lc, flat=flat,
        folded=fo, bat_flux=bf,
        best_period=bp, t0=t0, duration=dur,
        depth=dep, dur_hours=dur_h,
        snr=snr, rms=rms,
        signal_type=sig, confidence=conf,
        conf_lbl=conf_lbl, score=score,
        reasons=reasons, method=method,
        rp_rs=float(pm.rp),
        rp_earth=float(pm.rp)*109.2,
        fit_params=fit_params,
        star_props=star_props,
        sec_eclipse=sec_eclipse,
        planet2=planet2,
        next_transits=next_transits,
        min_per=min_per, max_per=max_per,
        raw_std=raw_std, clean_std=clean_std,
        noise_pct=noise_pct,
    )


def save_db(r):
    db_file = os.path.join(BASE_DIR, 'results_database.csv')
    row = {
        'DateTime'     : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'StarID'       : r['star_label'],
        'Method'       : r['method'],
        'SignalType'   : r['signal_type'],
        'Confidence'   : r['confidence'],
        'ConfLevel'    : r['conf_lbl'],
        'SNR'          : round(r['snr'],2),
        'Period_days'  : round(float(r['best_period'].value),4),
        'Duration_hrs' : round(r['dur_hours'],2),
        'Depth'        : round(r['depth'],6),
        'Rp_Rs'        : round(r['rp_rs'],4),
        'Rp_Earth'     : round(r['rp_earth'],1),
        'NoiseReduced' : round(r['noise_pct'],1),
    }
    db = (pd.concat([pd.read_csv(db_file), pd.DataFrame([row])], ignore_index=True)
          if os.path.exists(db_file) else pd.DataFrame([row]))
    db.to_csv(db_file, index=False)


def make_plotly(r):
    sig_clr = {'Planet Transit':GREEN,'Eclipsing Binary':YELLOW,
                'Starspot':PINK,'Noise':BLUE}.get(r['signal_type'],GREEN)
    lc=r['lc']; tr=r['trend_lc']; fl=r['flat']
    fo=r['folded']; d=r['depth']; dv=r['duration'].value

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            '1. Raw Light Curve & Starspot Trend',
            '2. Noise-Reduced Data vs BATMAN Model',
            '3. Phase-Folded Transit (zoomed)',
            f'4. Classification — {r["method"]}'
        ),
        vertical_spacing=0.09,
        row_heights=[0.25,0.25,0.25,0.25]
    )

    fig.add_trace(go.Scatter(
        x=lc.time.value, y=lc.flux.value, mode='markers',
        marker=dict(size=1.5,color=BLUE,opacity=0.6), name='Raw Flux',
        hovertemplate='<b>RAW</b><br>Time: %{x:.4f} d<br>Flux: %{y:.6f}<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tr.time.value, y=tr.flux.value, mode='lines',
        line=dict(color=PINK,width=2), name='Starspot Trend',
        hovertemplate='<b>TREND</b><br>Time: %{x:.4f} d<br>Flux: %{y:.6f}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fl.time.value, y=fl.flux.value, mode='markers',
        marker=dict(size=1.5,color=GREEN,opacity=0.6), name='Cleaned',
        hovertemplate=f'<b>CLEANED</b><br>Time: %{{x:.4f}} d<br>Flux: %{{y:.6f}}<br>Noise -{r["noise_pct"]:.1f}%<extra></extra>'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=fl.time.value, y=r['bat_flux'], mode='lines',
        line=dict(color=YELLOW,width=2.5), name='BATMAN',
        hovertemplate=f'<b>BATMAN</b><br>Time: %{{x:.4f}} d<br>Flux: %{{y:.6f}}<br>Period: {r["best_period"].value:.4f} d<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=fo.time.value, y=fo.flux.value, mode='markers',
        marker=dict(size=3,color=PURPLE,opacity=0.8), name='Phase Folded',
        hovertemplate=f'<b>FOLDED</b><br>Phase: %{{x:.4f}} d<br>Flux: %{{y:.6f}}<br>SNR: {r["snr"]:.2f}<extra></extra>'
    ), row=3, col=1)
    fig.add_hline(y=1-d, line_dash='dash', line_color=PINK, line_width=2,
                  annotation_text=f'Depth={d:.6f}',
                  annotation_font_color=PINK, row=3, col=1)
    fig.add_vline(x=0, line_dash='dot', line_color=YELLOW, line_width=2,
                  annotation_text='Transit Centre',
                  annotation_font_color=YELLOW, row=3, col=1)
    fig.update_xaxes(range=[-dv*3,dv*3], row=3, col=1)

    cats=list(r['score'].keys()); vals=list(r['score'].values())
    bclrs=[GREEN,PINK,YELLOW,BLUE]
    bclrs[vals.index(max(vals))]=sig_clr
    fig.add_trace(go.Bar(
        x=cats, y=vals, marker_color=bclrs,
        marker_line_color='white', marker_line_width=1.2,
        name='Scores', text=[f'{v}%' for v in vals],
        textposition='outside', textfont=dict(color='white',size=13),
        hovertemplate='<b>%{x}</b><br>Score: %{y}%<extra></extra>'
    ), row=4, col=1)

    fig.update_layout(
        title=dict(
            text=(
                f'<b>{r["star_label"]}</b> — '
                f'<span style="color:{sig_clr}">{r["signal_type"]}</span><br>'
                f'<span style="font-size:12px;color:{GRAY}">'
                f'Method: {r["method"]} | '
                f'Confidence: {r["confidence"]}% ({r["conf_lbl"]}) | '
                f'SNR: {r["snr"]:.2f} | '
                f'Period: {r["best_period"].value:.4f} d | '
                f'Noise -{r["noise_pct"]:.1f}%'
                f'</span>'
            ),
            font=dict(size=15,color='white'), x=0.5
        ),
        height=1400, paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color='white'),
        legend=dict(bgcolor=PANEL,bordercolor='rgba(91,123,224,0.2)',
                    borderwidth=1,font=dict(color='white')),
        hovermode='x unified'
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.07)',color=GRAY)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.07)',color=GRAY)
    return fig


# ─────────────────────────────────────────────────────────────
# TAB: DETECT
# ─────────────────────────────────────────────────────────────
with tab_detect:
    st.markdown(f"""
    <div style="margin-bottom:2rem">
      <div style="font-size:.72rem;letter-spacing:2px;color:{BLUE};
                  text-transform:uppercase;margin-bottom:.4rem">
        Transit Photometry Pipeline
      </div>
      <h1 style="font-size:2.2rem;font-weight:700;margin:0;line-height:1.2">
        Find planets hidden<br>in starlight.
      </h1>
      <p style="color:{GRAY};margin-top:.7rem;font-size:.95rem">
        Enter a TESS Star ID or upload your CSV.
        The AI pipeline automatically detects transit signals.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Model status banner
    if CNN_LOADED:
        st.markdown(f"""
        <div style="background:{GREEN}11;border:1px solid {GREEN}33;
                    border-radius:10px;padding:.8rem 1.2rem;margin-bottom:1rem;
                    font-size:.85rem;color:{GREEN}">
          ✅ <b>Real CNN Model Active</b> —
          Classifications are ML-based (not hardcoded rules).
          Run <code>python train_classifier.py</code> to retrain.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{YELLOW}11;border:1px solid {YELLOW}33;
                    border-radius:10px;padding:.8rem 1.2rem;margin-bottom:1rem;
                    font-size:.85rem;color:{YELLOW}">
          ⚠️ <b>CNN not found</b> — using rule-based fallback.
          Run <code>python train_classifier.py</code> to train the real CNN model.
        </div>
        """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2,1], gap='large')

    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="input-label">TESS Star ID</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="input-desc">
          Enter the TIC number from NASA's TESS Input Catalog.<br>
          <b style="color:white">Example:</b> TIC 261136679 (confirmed exoplanet host)
        </div>
        """, unsafe_allow_html=True)

        star_id = st.text_input('Star ID', value='TIC 261136679',
                                label_visibility='collapsed')
        st.markdown(f'<div class="input-label" style="margin-top:1rem">TESS Sector (0–10)</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="input-desc">Each sector = 27 days of data. Sector 1 works best for this star.</div>',
                    unsafe_allow_html=True)
        sector   = st.slider('Sector', 0, 10, 1, label_visibility='collapsed')
        run_tess = st.button('🔍  Analyse Star', key='rt')

        st.markdown('<hr style="border-color:#ffffff11;margin:1.5rem 0">', unsafe_allow_html=True)
        st.markdown(f'<div class="input-label">Or Upload CSV</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="input-desc">
          CSV must have columns: <b style="color:white">TIME</b>,
          <b style="color:white">PDCSAP_FLUX</b>, QUALITY (optional).<br>
          Download from: <b style="color:{BLUE}">archive.stsci.edu/tess</b>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader('CSV', type=['csv'], label_visibility='collapsed')
        run_csv  = st.button('🔍  Analyse CSV', key='rc')
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div class="glass-card">
          <div class="input-label">How it works</div>
          <div class="step-item">
            <div class="step-num">1</div>
            <div><b>Download</b> — Fetch TESS light curve from NASA archive</div>
          </div>
          <div class="step-item">
            <div class="step-num">2</div>
            <div><b>Denoise</b> — Savitzky-Golay filter + 3-layer sigma clipping</div>
          </div>
          <div class="step-item">
            <div class="step-num">3</div>
            <div><b>BLS Detect</b> — Box Least Squares finds periodic dips</div>
          </div>
          <div class="step-item">
            <div class="step-num">4</div>
            <div><b>CNN Classify</b> — Neural network classifies signal type</div>
          </div>
          <div class="step-item">
            <div class="step-num">5</div>
            <div><b>BATMAN Fit</b> — Estimates Period, Depth, Duration</div>
          </div>
          <div class="step-item" style="border-bottom:none">
            <div class="step-num">6</div>
            <div><b>Results</b> — Interactive graph + confidence + database</div>
          </div>
        </div>
        <div class="wait-box">
          ⏳ <b>Processing takes 2–4 minutes.</b><br>
          Do not refresh. When done click <b>📊 Graph</b> tab!
        </div>
        """, unsafe_allow_html=True)

    def show_result_card(r):
        sig_clr = {'Planet Transit':GREEN,'Eclipsing Binary':YELLOW,
                   'Starspot':PINK,'Noise':BLUE}.get(r['signal_type'],GREEN)
        st.markdown(f"""
        <div class="success-box">
          Detection complete! ✅  Result: <b>{r['signal_type']}</b>
          with <b>{r['confidence']}% {r['conf_lbl']} confidence</b>
          using <b>{r['method']}</b> classification.<br>
          Click <b>📊 Graph</b> tab to see full interactive analysis!
        </div>
        <div class="glass-card" style="border-color:{sig_clr}44">
          <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
            <span class="result-badge"
              style="background:{sig_clr}22;border:1px solid {sig_clr};color:{sig_clr}">
              {r['signal_type']}
            </span>
            <span style="color:{GRAY};font-size:.85rem">{r['star_label']}</span>
            <span style="color:{GRAY};font-size:.75rem;margin-left:auto">via {r['method']}</span>
          </div>
          <div class="metric-row">
            <div class="metric-box"><div class="metric-label">Confidence</div>
              <div class="metric-value" style="color:{sig_clr}">{r['confidence']}%</div></div>
            <div class="metric-box"><div class="metric-label">SNR</div>
              <div class="metric-value">{r['snr']:.2f}</div></div>
            <div class="metric-box"><div class="metric-label">Period</div>
              <div class="metric-value">{r['best_period'].value:.4f} d</div></div>
            <div class="metric-box"><div class="metric-label">Duration</div>
              <div class="metric-value">{r['dur_hours']:.2f} h</div></div>
          </div>
          <div class="metric-row">
            <div class="metric-box"><div class="metric-label">Depth</div>
              <div class="metric-value">{r['depth']:.6f}</div></div>
            <div class="metric-box"><div class="metric-label">Planet Radius</div>
              <div class="metric-value">~{r['rp_earth']:.1f} R⊕</div></div>
            <div class="metric-box"><div class="metric-label">Noise Reduced</div>
              <div class="metric-value" style="color:{GREEN}">{r['noise_pct']:.1f}%</div></div>
            <div class="metric-box"><div class="metric-label">Conf. Level</div>
              <div class="metric-value">{r['conf_lbl']}</div></div>
          </div>
          <div class="section-title">Why this classification?</div>
          {''.join(f'<div class="reason-item">{rr}</div>' for rr in r["reasons"])}
        </div>
        """, unsafe_allow_html=True)

        fp = r.get('fit_params', {})
        if fp:
            st.markdown(f"""
            <div class="glass-card">
              <div class="input-label">BATMAN Fitted Parameters (scipy optimize)</div>
              <div class="metric-row">
                <div class="metric-box"><div class="metric-label">Rp/Rs</div>
                  <div class="metric-value">{fp['rp']}</div></div>
                <div class="metric-box"><div class="metric-label">Semi-major axis</div>
                  <div class="metric-value">{fp['a']} R★</div></div>
                <div class="metric-box"><div class="metric-label">Inclination</div>
                  <div class="metric-value">{fp['inc']}°</div></div>
                <div class="metric-box"><div class="metric-label">R² Fit</div>
                  <div class="metric-value" style="color:{GREEN}">{fp['r_squared']}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    if run_tess:
        with st.status('Running pipeline... ⏳', expanded=True) as status:
            st.write('📡 Downloading TESS data...')
            try:
                st.write('🧹 Removing noise (Savitzky-Golay + sigma clipping)...')
                st.write('📦 Running BLS transit detection...')
                st.write('🧠 Classifying signal...')
                st.write('🪐 Fitting BATMAN transit model...')
                r = run_pipeline(False, None, star_id, sector)
                save_db(r)
                st.session_state.results = r
                status.update(label='Done! ✅', state='complete')
            except Exception as e:
                status.update(label='Error!', state='error')
                st.error(f'Error: {e}')
        if st.session_state.results:
            show_result_card(st.session_state.results)

    if run_csv and uploaded:
        df_up = pd.read_csv(uploaded)
        with st.status('Running pipeline on CSV... ⏳', expanded=True) as status:
            st.write('🧹 Removing noise...')
            st.write('📦 Running BLS...')
            st.write('🧠 Classifying...')
            try:
                r = run_pipeline(True, df_up, '', 0)
                save_db(r)
                st.session_state.results = r
                status.update(label='Done! ✅', state='complete')
            except Exception as e:
                status.update(label='Error!', state='error')
                st.error(f'Error: {e}')
        if st.session_state.results:
            show_result_card(st.session_state.results)


# ─────────────────────────────────────────────────────────────
# TAB: GRAPH
# ─────────────────────────────────────────────────────────────
with tab_graph:
    st.markdown('<h2 style="margin-bottom:1.5rem">Interactive Light Curve Analysis</h2>',
                unsafe_allow_html=True)
    if not st.session_state.results:
        st.markdown(f"""
        <div class="wait-box" style="text-align:center;padding:2rem">
          No results yet.<br><br>
          Go to <b>🔭 Detect</b> → enter Star ID → click <b>Analyse Star</b><br>
          Then come back here!
        </div>
        """, unsafe_allow_html=True)
    else:
        r   = st.session_state.results
        fig = make_plotly(r)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            st.download_button('⬇️ Interactive HTML',
                               fig.to_html(config={'scrollZoom':True}).encode(),
                               'exoplanet_graph.html','text/html')
        with c2:
            try:
                st.download_button('⬇️ PNG Image',
                                   fig.to_image(format='png',width=1600,height=1400),
                                   'exoplanet_graph.png','image/png')
            except:
                st.info('pip install kaleido for PNG export')


# ─────────────────────────────────────────────────────────────
# TAB: DATABASE
# ─────────────────────────────────────────────────────────────
with tab_db:
    st.markdown('<h2 style="margin-bottom:1.5rem">Detection Database</h2>',
                unsafe_allow_html=True)
    db_file = os.path.join(BASE_DIR, 'results_database.csv')
    if os.path.exists(db_file):
        db = pd.read_csv(db_file)
        planets  = len(db[db['SignalType']=='Planet Transit'])
        binaries = len(db[db['SignalType']=='Eclipsing Binary'])
        noise    = len(db[db['SignalType']=='Noise'])
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="metric-label">Total</div>
            <div class="metric-value">{len(db)}</div></div>
          <div class="metric-box"><div class="metric-label">Planets</div>
            <div class="metric-value" style="color:{GREEN}">{planets}</div></div>
          <div class="metric-box"><div class="metric-label">Binaries</div>
            <div class="metric-value" style="color:{YELLOW}">{binaries}</div></div>
          <div class="metric-box"><div class="metric-label">Noise</div>
            <div class="metric-value" style="color:{GRAY}">{noise}</div></div>
        </div>
        """, unsafe_allow_html=True)
        filt = st.selectbox('Filter',['All']+list(db['SignalType'].unique()))
        show = db if filt=='All' else db[db['SignalType']==filt]
        st.dataframe(show, use_container_width=True, height=400)
        st.download_button('⬇️ Download CSV',
                           db.to_csv(index=False).encode(),
                           'results.csv','text/csv')

        if len(db) >= 2:
            st.markdown("---")
            st.markdown('<h3>Period vs Planet Radius</h3>', unsafe_allow_html=True)
            st.plotly_chart(make_database_chart(db), use_container_width=True)
    else:
        st.markdown(f'<div class="wait-box">No entries yet. Run a detection first!</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB: ARCHITECTURE
# ─────────────────────────────────────────────────────────────
with tab_arch:
    st.markdown('<h2 style="margin-bottom:.5rem">Pipeline Architecture</h2>',
                unsafe_allow_html=True)
    st.markdown(f'<p style="color:{GRAY};margin-bottom:2rem">End-to-end AI transit detection system.</p>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1,1], gap='large')

    with col_a:
        steps = [
            ('🛰️','TESS / CSV Data','NASA MAST · 20–30k light curves','INPUT',BLUE),
            ('🧹','Preprocessing','Lightkurve · Savitzky-Golay · σ-clip','STEP 1',GREEN),
            ('📦','BLS Detection','Astropy BLS · Period · Depth · Duration','STEP 2',PURPLE),
            ('🧠','CNN Classification','Conv1D x3 → Dense → Softmax (4 classes)','STEP 3',YELLOW),
            ('🚫','False +ve Rejection','XGBoost · Removes binaries & noise','STEP 4',PINK),
            ('🪐','BATMAN Fitting','Transit model · Parameter estimation','STEP 5',BLUE),
            ('📊','Visualization','Plotly interactive · Phase fold','OUTPUT',GREEN),
        ]
        for i,(icon,title,sub,badge,clr) in enumerate(steps):
            st.markdown(f"""
            <div class="arch-node" style="border-color:{clr}55">
              <div style="font-size:1.3rem">{icon}</div>
              <div style="flex:1">
                <div style="font-weight:700;font-size:.95rem">{title}</div>
                <div style="font-size:.75rem;color:{GRAY}">{sub}</div>
              </div>
              <div class="arch-badge"
                style="background:{clr}11;border:1px solid {clr}55;color:{clr}">
                {badge}
              </div>
            </div>
            {"<div class='arch-arrow'>↓</div>" if i<len(steps)-1 else ""}
            """, unsafe_allow_html=True)

    with col_b:
        # Accuracy metrics (real after training)
        if os.path.exists(META_PATH):
            meta = pd.read_csv(META_PATH)
            real_acc = meta['test_accuracy'].iloc[0]
            st.markdown(f"""
            <div class="glass-card" style="border-color:{GREEN}44;margin-bottom:1rem">
              <div class="input-label">Real Model Performance</div>
              <div class="metric-row" style="grid-template-columns:repeat(2,1fr)">
                <div class="metric-box"><div class="metric-label">Test Accuracy</div>
                  <div class="metric-value" style="color:{GREEN}">{real_acc}%</div></div>
                <div class="metric-box"><div class="metric-label">Classes</div>
                  <div class="metric-value" style="font-size:.85rem">4</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div class="section-title">Technology Stack</div>',
                    unsafe_allow_html=True)
        stack=[
            ('Data','Lightkurve + MAST','NASA official'),
            ('Noise','Savitzky-Golay + σ-clip','3-layer pipeline'),
            ('Detection','Astropy BLS','Physics-based'),
            ('ML Model','TensorFlow CNN','Conv1D × 3'),
            ('FP Filter','XGBoost','Rule-augmented'),
            ('Fitting','BATMAN','NASA standard'),
            ('Viz','Plotly','Interactive'),
            ('DB','CSV','Auto-logged'),
        ]
        for comp,tool,note in stack:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:9px 14px;margin:4px 0;border-radius:8px;
                        background:{BG};border:1px solid {BLUE}22">
              <div>
                <div style="font-size:.72rem;color:{GRAY}">{comp}</div>
                <div style="font-weight:600;font-size:.9rem">{tool}</div>
              </div>
              <div style="font-size:.72rem;color:{BLUE}">{note}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div class="section-title">Novelty Points</div>',
                    unsafe_allow_html=True)
        novelties=[
            (GREEN,'Real CNN Classifier','Not hardcoded rules — trained on data'),
            (YELLOW,'4-Class Output','Planet / Binary / Starspot / Noise'),
            (PINK,'Physics + AI Hybrid','BLS detection → CNN classification'),
            (PURPLE,'vs ExoMiner++','Free laptop vs NASA supercomputer'),
        ]
        for clr,title,desc in novelties:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;
                        padding:12px 14px;margin:5px 0;border-radius:10px;
                        background:{clr}0d;border:1px solid {clr}33">
              <div style="width:8px;height:8px;border-radius:50%;
                          background:{clr};margin-top:5px;flex-shrink:0"></div>
              <div>
                <div style="font-weight:700;font-size:.9rem">{title}</div>
                <div style="font-size:.78rem;color:{GRAY};margin-top:2px">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB: 3D ORBIT
# ─────────────────────────────────────────────────────────────
with tab_3d:
    st.markdown('<h2 style="margin-bottom:1rem">3D Planetary Orbit Viewer</h2>',
                unsafe_allow_html=True)

    if not st.session_state.results:
        st.markdown(f"""
        <div class="wait-box" style="text-align:center;padding:2rem">
          No results yet.<br><br>
          Go to <b>🔭 Detect</b> → enter Star ID → click <b>Analyse Star</b><br>
          Then come back here!
        </div>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state.results
        inc = r.get('fit_params', {}).get('inc', 87.0)

        fig3d = make_3d_orbit(
            period_days=float(r['best_period'].value),
            rp_rs=r['rp_rs'],
            inclination_deg=inc,
            signal_type=r['signal_type']
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box">
            <div class="metric-label">Orbital Period</div>
            <div class="metric-value">{float(r['best_period'].value):.4f} d</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Planet Radius</div>
            <div class="metric-value">{r['rp_rs']:.4f} Rp/Rs</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Inclination</div>
            <div class="metric-value">{inc:.1f}°</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Signal Type</div>
            <div class="metric-value" style="font-size:.9rem">{r['signal_type']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            '⬇️ Download 3D View (HTML)',
            fig3d.to_html(config={'scrollZoom': True}).encode(),
            'orbit_3d.html',
            'text/html'
        )


# ─────────────────────────────────────────────────────────────
# TAB: AI ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab_ai:
    st.markdown('<h2 style="margin-bottom:.5rem">🤖 AI Analysis Suite</h2>',
                unsafe_allow_html=True)

    if not st.session_state.results:
        st.markdown(f"""
        <div class="wait-box" style="text-align:center;padding:2rem">
          No results yet — run a detection first in the 🔭 Detect tab!
        </div>
        """, unsafe_allow_html=True)
    elif not ANTHROPIC_SDK_OK:
        st.error("⚠️ The `anthropic` package isn't installed. Run `pip install anthropic` to enable this tab.")
    elif not ANTHROPIC_API_KEY:
        st.error("⚠️ No Anthropic API key found. Add ANTHROPIC_API_KEY to your environment or st.secrets.")
    else:
        r = st.session_state.results
        context = build_planet_context(r)

        ai_tab1, ai_tab2, ai_tab3 = st.tabs([
            "🤖 Exoplanet Chatbot", "📄 Scientific Report", "🌱 Habitability Analysis"
        ])

        # ── CHATBOT ───────────────────────────────────
        with ai_tab1:
            st.markdown(f"""
            <div style="background:{PANEL};border:1px solid {BLUE}33;
                        border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem">
              <div style="font-size:.75rem;color:{GRAY};margin-bottom:.3rem">
                ABOUT THIS PLANET
              </div>
              <div style="font-size:.85rem;color:white">
                Ask me anything about <b>{r['star_label']}</b> —
                the detected <b>{r['signal_type']}</b> candidate.
              </div>
            </div>
            """, unsafe_allow_html=True)

            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            for msg in st.session_state.chat_history:
                role  = msg['role']
                color = GREEN if role == 'assistant' else BLUE
                icon  = '🤖' if role == 'assistant' else '👤'
                st.markdown(f"""
                <div style="display:flex;gap:10px;margin:8px 0;
                            {'flex-direction:row-reverse' if role=='user' else ''}">
                  <div style="font-size:1.2rem">{icon}</div>
                  <div style="background:{PANEL};border:1px solid {color}33;
                              border-radius:10px;padding:10px 14px;
                              max-width:80%;font-size:.88rem;color:white;
                              line-height:1.6">
                    {msg['content']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            suggestions = [
                "Is this planet in the habitable zone?",
                "How does this compare to Earth?",
                "Could this be a false positive?",
                "What follow-up observations are needed?",
            ]
            st.markdown(f'<div style="font-size:.72rem;color:{GRAY};margin:.5rem 0">Quick questions:</div>',
                        unsafe_allow_html=True)
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(suggestion, key=f'sug_{i}'):
                    with st.spinner('ExoAI is thinking...'):
                        reply = ai_chatbot_response(suggestion, context, st.session_state.chat_history)
                    st.session_state.chat_history.append({'role': 'user', 'content': suggestion})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
                    st.rerun()

            user_q = st.text_input('Ask ExoAI anything about this planet...',
                                   key='chat_input',
                                   placeholder='e.g. What type of star is this?')
            c1, c2 = st.columns([3, 1])
            with c2:
                if st.button('Send 🚀', key='send_chat') and user_q:
                    with st.spinner('ExoAI is thinking...'):
                        reply = ai_chatbot_response(user_q, context, st.session_state.chat_history)
                    st.session_state.chat_history.append({'role': 'user', 'content': user_q})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
                    st.rerun()
            with c1:
                if st.button('🗑️ Clear Chat', key='clear_chat'):
                    st.session_state.chat_history = []
                    st.rerun()

        # ── SCIENTIFIC REPORT ─────────────────────────
        with ai_tab2:
            st.markdown(f"""
            <div class="glass-card">
              <div class="input-label">AI Scientific Report</div>
              <div class="input-desc">
                Claude generates a peer-review style report based on your
                detection results. Takes ~10 seconds.
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button('📄 Generate Scientific Report', key='gen_report'):
                with st.spinner('Claude is writing your report...'):
                    report = generate_ai_report(context, r['star_label'])
                st.session_state['ai_report'] = report

            if 'ai_report' in st.session_state:
                st.markdown(st.session_state['ai_report'])
                st.download_button(
                    '⬇️ Download Report (.md)',
                    st.session_state['ai_report'].encode(),
                    f"report_{r['star_label'].replace(' ','_')}.md",
                    'text/markdown'
                )

        # ── HABITABILITY ──────────────────────────────
        with ai_tab3:
            st.markdown(f"""
            <div class="glass-card">
              <div class="input-label">AI Habitability Analysis</div>
              <div class="input-desc">
                Claude analyzes orbital parameters, planet size, and transit
                data to estimate the planet's potential for supporting life.
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button('🌱 Analyse Habitability', key='gen_hab'):
                with st.spinner('Analysing habitability...'):
                    hab = analyze_habitability(context)
                st.session_state['hab_result'] = hab

            if 'hab_result' in st.session_state:
                hab   = st.session_state['hab_result']
                score = hab.get('score', 0)
                clr   = GREEN if score >= 60 else YELLOW if score >= 30 else PINK

                st.markdown(f"""
                <div class="glass-card" style="border-color:{clr}44">
                  <div style="display:flex;align-items:center;
                              justify-content:space-between;margin-bottom:1.2rem">
                    <div>
                      <div style="font-size:2.5rem;font-weight:800;
                                  color:{clr};font-family:'JetBrains Mono',monospace">
                        {score}/100
                      </div>
                      <div style="font-size:1rem;color:white;font-weight:600">
                        {hab.get('verdict','Unknown')}
                      </div>
                      <div style="font-size:.8rem;color:{GRAY}">
                        Zone: {hab.get('zone','Unknown')}
                      </div>
                    </div>
                    <div style="text-align:right">
                      <div style="font-size:.72rem;color:{GRAY}">Similar to</div>
                      <div style="font-size:1rem;color:{BLUE};font-weight:600">
                        {hab.get('similar_planet','Unknown')}
                      </div>
                    </div>
                  </div>

                  <div style="background:{BG};border-radius:10px;
                              height:12px;margin-bottom:1.2rem;overflow:hidden">
                    <div style="width:{score}%;height:100%;
                                background:linear-gradient(90deg,{clr},{clr}88);
                                border-radius:10px;transition:width .5s">
                    </div>
                  </div>

                  <div class="input-label">Key Factors</div>
                  {"".join(f'<div class="reason-item">✦ {f}</div>' for f in hab.get('factors', []))}

                  <div class="input-label" style="margin-top:1rem">AI Explanation</div>
                  <div style="font-size:.88rem;color:#ccc;line-height:1.7;
                              background:{BG};border-radius:8px;padding:12px">
                    {hab.get('explanation','')}
                  </div>

                  <div class="input-label" style="margin-top:1rem">
                    Recommended Follow-up
                  </div>
                  <div style="font-size:.88rem;color:{BLUE};
                              background:{BLUE}11;border-radius:8px;
                              padding:10px 14px;border-left:3px solid {BLUE}">
                    🔭 {hab.get('follow_up','')}
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB: DEEP ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab_extra:
    st.markdown('<h2 style="margin-bottom:1rem">🔬 Deep Analysis</h2>',
                unsafe_allow_html=True)

    if not st.session_state.results:
        st.markdown('''
        <div class="wait-box" style="text-align:center;padding:2rem">
          No results yet — run a detection first in 🔭 Detect tab!
        </div>
        ''', unsafe_allow_html=True)
    else:
        r = st.session_state.results

        da1, da2, da3, da4 = st.tabs([
            "⭐ Star Properties", "🔴 Secondary Eclipse",
            "🪐 Multi-Planet", "📅 Transit Alerts"
        ])

        # ── STAR PROPERTIES ───────────────────────────────────
        with da1:
            sp = r.get('star_props', {})
            found = sp.get('found', False)
            source_label = 'NASA TIC Catalog' if found else 'Default (Solar)'
            badge_clr = GREEN if found else YELLOW

            st.markdown(f'''
            <div class="glass-card">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:1rem">
                <div class="input-label">Host Star Properties</div>
                <span style="font-size:.72rem;padding:4px 12px;border-radius:20px;
                             background:{badge_clr}22;color:{badge_clr};
                             border:1px solid {badge_clr}55">
                  {source_label}
                </span>
              </div>
              <div class="metric-row">
                <div class="metric-box">
                  <div class="metric-label">Radius</div>
                  <div class="metric-value">{sp.get("radius_solar",1.0):.2f} R☉</div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">Temperature</div>
                  <div class="metric-value">{sp.get("temp_k",5778):.0f} K</div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">Mass</div>
                  <div class="metric-value">{sp.get("mass_solar",1.0):.2f} M☉</div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">log g</div>
                  <div class="metric-value">{sp.get("logg",4.44):.2f}</div>
                </div>
              </div>
            </div>
            ''', unsafe_allow_html=True)

            if sp.get('radius_solar'):
                rp_real = r['rp_rs'] * sp['radius_solar'] * 109.2
                st.markdown(f'''
                <div class="glass-card" style="border-color:{GREEN}44">
                  <div class="input-label">Corrected Planet Radius</div>
                  <div style="font-size:2rem;font-weight:800;
                              color:{GREEN};font-family:JetBrains Mono,monospace">
                    {rp_real:.2f} R⊕
                  </div>
                  <div style="font-size:.8rem;color:{GRAY};margin-top:.3rem">
                    Using real star radius {sp["radius_solar"]:.2f} R☉
                    (vs assumed 1.0 R☉ without TIC data)
                  </div>
                </div>
                ''', unsafe_allow_html=True)

        # ── SECONDARY ECLIPSE ─────────────────────────────────
        with da2:
            sec = r.get('sec_eclipse', {})
            has_sec = sec.get('has_secondary', False)
            sec_clr = PINK if has_sec else GREEN

            st.markdown(f'''
            <div class="glass-card" style="border-color:{sec_clr}44">
              <div class="input-label">Secondary Eclipse Check</div>
              <div style="font-size:1rem;font-weight:700;
                          color:{sec_clr};margin:1rem 0">
                {sec.get("verdict","No data")}
              </div>
              <div class="metric-row">
                <div class="metric-box">
                  <div class="metric-label">Secondary Depth</div>
                  <div class="metric-value">
                    {sec.get("secondary_depth", 0.0):.6f}
                  </div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">Depth Ratio</div>
                  <div class="metric-value">
                    {sec.get("depth_ratio", 0.0):.4f}
                  </div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">False Positive Risk</div>
                  <div class="metric-value" style="color:{sec_clr}">
                    {"HIGH" if has_sec else "LOW"}
                  </div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">Likely Signal</div>
                  <div class="metric-value" style="font-size:.85rem">
                    {"Eclipsing Binary" if has_sec else "Planet Transit"}
                  </div>
                </div>
              </div>
              <div style="font-size:.82rem;color:{GRAY};margin-top:.5rem">
                A secondary eclipse at orbital phase 0.5 indicates the companion
                is self-luminous (eclipsing binary), not a dark planet.
                Ratio > 0.1 is suspicious. Ratio > 0.5 = almost certainly not a planet.
              </div>
            </div>
            ''', unsafe_allow_html=True)

        # ── MULTI-PLANET ──────────────────────────────────────
        with da3:
            p2 = r.get('planet2', {})
            if p2.get('found'):
                st.markdown(f'''
                <div class="glass-card" style="border-color:{PURPLE}44">
                  <div class="input-label">🎉 Second Planet Candidate Detected!</div>
                  <div class="metric-row">
                    <div class="metric-box">
                      <div class="metric-label">Period</div>
                      <div class="metric-value"
                           style="color:{PURPLE}">{p2["period"]} d</div>
                    </div>
                    <div class="metric-box">
                      <div class="metric-label">Transit Depth</div>
                      <div class="metric-value">{p2["depth"]:.6f}</div>
                    </div>
                    <div class="metric-box">
                      <div class="metric-label">Duration</div>
                      <div class="metric-value">{p2["duration_hrs"]:.2f} h</div>
                    </div>
                    <div class="metric-box">
                      <div class="metric-label">SNR</div>
                      <div class="metric-value"
                           style="color:{GREEN}">{p2["snr"]}</div>
                    </div>
                  </div>
                  <div class="metric-row">
                    <div class="metric-box">
                      <div class="metric-label">Rp/Rs</div>
                      <div class="metric-value">{p2["rp_rs"]}</div>
                    </div>
                    <div class="metric-box">
                      <div class="metric-label">Planet Radius</div>
                      <div class="metric-value">~{p2["rp_earth"]} R⊕</div>
                    </div>
                  </div>
                  <div style="background:{PURPLE}11;border-left:3px solid {PURPLE};
                              border-radius:0 8px 8px 0;padding:10px 14px;
                              font-size:.85rem;color:#ccc;margin-top:.5rem">
                    Method: Primary transit signal removed → BLS re-run on residual
                    light curve → Secondary period found with SNR {p2["snr"]} ≥ 3.
                  </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="glass-card">
                  <div class="input-label">Multi-Planet Search</div>
                  <div style="color:{GRAY};font-size:.9rem;padding:1rem 0">
                    No second planet candidate found with SNR ≥ 3 after
                    removing the primary transit signal. This star may host
                    only one transiting planet, or a second planet may transit
                    at a period outside the search range
                    ({r.get("min_per",0.5):.1f}–{r.get("max_per",12):.1f} days).
                  </div>
                </div>
                ''', unsafe_allow_html=True)

        # ── TRANSIT ALERTS ────────────────────────────────────
        with da4:
            transits = r.get('next_transits', [])
            period   = float(r['best_period'].value)
            dur_h    = r.get('dur_hours', 0)

            st.markdown(f'''
            <div class="glass-card">
              <div class="input-label">Next 5 Predicted Transits</div>
              <div style="font-size:.82rem;color:{GRAY};margin-bottom:1rem">
                Based on detected period {period:.4f} d and transit epoch.
                Times are approximate (UTC). Observe ±{dur_h/2:.1f}h around mid-time.
              </div>
            ''', unsafe_allow_html=True)

            for i, tr in enumerate(transits):
                days = tr['days_from_now']
                urgency_clr = GREEN if days < 7 else BLUE if days < 30 else GRAY
                st.markdown(f'''
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;padding:12px 16px;margin:6px 0;
                              background:{BG};border-radius:10px;
                              border:1px solid {urgency_clr}44">
                    <div>
                      <div style="font-size:.72rem;color:{GRAY}">
                        Transit #{tr["transit_num"]}
                      </div>
                      <div style="font-weight:700;color:white;font-size:.95rem">
                        {tr["utc_date"]}
                      </div>
                    </div>
                    <div style="text-align:right">
                      <div style="font-size:.72rem;color:{GRAY}">Days away</div>
                      <div style="font-size:1.2rem;font-weight:800;
                                  color:{urgency_clr};
                                  font-family:JetBrains Mono,monospace">
                        {days:.1f}
                      </div>
                    </div>
                  </div>
                ''', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)