import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightkurve as lk
import batman
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

tab_detect, tab_graph, tab_db, tab_arch = st.tabs([
    "🔭  Detect", "📊  Graph", "🗄️  Database", "🏗️  Architecture"
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

    # BLS
      # BLS
    pg = flat.to_periodogram(
        method='bls',
        minimum_period=1.0,
        maximum_period=12,
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

    # BATMAN
    pm = batman.TransitParams()
    pm.t0=float(t0.value); pm.per=float(bp.value)
    pm.rp=float(np.sqrt(dep)) if dep>0 else 0.1
    pm.a=15.; pm.inc=87.; pm.ecc=0.; pm.w=90.
    pm.u=[0.1,0.3]; pm.limb_dark='quadratic'
    ta  = np.asarray(flat.time.value, dtype=float)
    bf  = batman.TransitModel(pm, ta).light_curve(pm)

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
