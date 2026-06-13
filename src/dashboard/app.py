import os
import sys
from pathlib import Path

# Add project root to python path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import time
import threading
import numpy as np
import queue

try:
    import streamlit as st
    import plotly.graph_objects as go
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    from src.inference.realtime import WebRTCVideoProcessor
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    webrtc_streamer = None
    RTCConfiguration = None
    WebRTCVideoProcessor = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCORE_COLORS = {
    'low': '#27ae60',      # Green — alert
    'medium': '#f39c12',   # Amber — mild fatigue
    'high': '#e74c3c',     # Red — high fatigue
}


def score_color(score: float) -> str:
    if score < 40:
        return SCORE_COLORS['low']
    elif score < 70:
        return SCORE_COLORS['medium']
    return SCORE_COLORS['high']


def score_label(score: float) -> str:
    if score < 40:
        return 'Alert'
    elif score < 70:
        return 'Mild Fatigue'
    return 'High Fatigue'


def alert_emoji(alert: str) -> str:
    if alert == 'FATIGUE_ONSET':
        return '⚠️ FATIGUE ONSET'
    elif alert == 'RECOVERY':
        return '✅ RECOVERY'
    return 'Stable'


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def run_app():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Run: pip install streamlit plotly")
        return

    st.set_page_config(
        layout='wide',
        page_title='Fatigue Drift Monitor',
        page_icon='🧠',
    )

    # --- Inject Custom CSS for Premium Styling ---
    st.markdown("""
    <style>
        /* Modern Premium CSS styling */
        .stApp {
            background-color: #0f111a;
            color: #ecf0f1;
        }
        h1, h2, h3 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }
        .main-header {
            text-align: center;
            padding: 10px 0 20px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 20px;
        }
        div[data-testid="stSidebar"] {
            background-color: #161925;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        /* Custom styled card containers */
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stContainer"]) {
            background: rgba(22, 25, 37, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(8px);
        }
        .connection-status {
            font-size: 14px;
            font-weight: 500;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Session State Init ---
    if 'score_history' not in st.session_state:
        st.session_state.score_history = []
        st.session_state.time_history = []
        st.session_state.confidence_history = []
        st.session_state.alert_history = []
        st.session_state.attributions_history = []
        st.session_state.session_active = False
        st.session_state.session_start = None
        st.session_state.pipeline_stop = None
        st.session_state.score_queue = None
        st.session_state.video_queue = None
        st.session_state.capture_thread = None
        st.session_state.calibration_toast_shown = False

    # --- Sidebar ---
    with st.sidebar:
        st.title('🧠 Session Controls')
        st.markdown('---')
        user_id = st.text_input('User ID', value='user_001')

        col_start, col_stop = st.columns(2)
        start_btn = col_start.button('▶️ Start', use_container_width=True)
        stop_btn = col_stop.button('⏹ Stop', use_container_width=True)

        st.markdown('---')
        st.subheader('Settings')
        simulate = st.checkbox('Simulation Mode', value=False,
                               help='Generate synthetic data instead of webcam')
        camera_id = st.number_input('Camera ID', min_value=0, max_value=5, value=0, step=1,
                                    help='OpenCV camera index (0 for built-in, 1+ for external)')
        refresh_rate = st.slider('Refresh (sec)', 1, 10, 3)

        st.markdown('---')
        st.caption('v1.0 • Fatigue Drift Monitor')
        st.caption('RTX 3050 Laptop GPU')

    # --- Handle Start / Stop ---
    if start_btn and not st.session_state.session_active:
        st.session_state.score_history = []
        st.session_state.time_history = []
        st.session_state.confidence_history = []
        st.session_state.alert_history = []
        st.session_state.attributions_history = []
        st.session_state.calibration_toast_shown = False
        st.session_state.session_active = True
        st.session_state.session_start = time.time()

        # Force-reload modules to apply code updates without restarting the Streamlit server
        import importlib
        import src.models.cusum
        import src.extraction.face_mesh
        import src.explainability.attribution
        import src.inference.realtime
        
        importlib.reload(src.models.cusum)
        importlib.reload(src.extraction.face_mesh)
        importlib.reload(src.explainability.attribution)
        importlib.reload(src.inference.realtime)
        
        from src.inference.realtime import start_pipeline
        from src.models.cusum import CUSUMDetector
        
        encoder = None
        drift_model = None
        if not simulate:
            import torch
            from pathlib import Path
            from src.models.short_encoder import ShortTermEncoder
            from src.models.drift_model import DriftModel
            
            # Find checkpoint
            ckpt_files = list(Path('models').glob('*.ckpt'))
            dmd_ckpts = [c for c in ckpt_files if 'dmd' in c.name]
            if dmd_ckpts:
                checkpoint_path = sorted(dmd_ckpts)[-1]
            elif ckpt_files:
                checkpoint_path = sorted(ckpt_files)[-1]
            else:
                checkpoint_path = None
                
            if checkpoint_path and Path(checkpoint_path).exists():
                try:
                    ckpt = torch.load(checkpoint_path, map_location='cpu')
                    state_dict = ckpt.get('state_dict', ckpt)
                    
                    # Discover hyperparameters from state dict dynamically
                    from scripts.export_onnx import discover_hyperparameters, clean_state_dict
                    hparams = discover_hyperparameters(state_dict)
                    encoder = ShortTermEncoder(
                        input_dim=hparams['input_dim'],
                        hidden_dim=hparams['hidden_dim'],
                        num_layers=hparams['num_layers'],
                        output_dim=hparams['output_dim']
                    )
                    prefix = "encoder."
                    if not any(k.startswith(prefix) for k in state_dict.keys()):
                        if any(k.startswith("model.encoder.") for k in state_dict.keys()):
                            prefix = "model.encoder."
                    cleaned_sd = clean_state_dict(state_dict, prefix=prefix)
                    encoder.load_state_dict(cleaned_sd, strict=True)
                    encoder.eval()
                    st.toast(f"Loaded encoder from {checkpoint_path.name}! 🎯")
                except Exception as e:
                    st.warning(f"Error loading checkpoint weights: {e}. Falling back to default architecture.")
                    encoder = ShortTermEncoder()
            else:
                st.warning("No checkpoint found. Running with uninitialized encoder.")
                encoder = ShortTermEncoder()
                
            drift_model = DriftModel()
            
        st.session_state.video_queue = queue.Queue(maxsize=10)
        st.session_state.raw_features_queue = queue.Queue(maxsize=100) if not simulate else None
        
        stop_event, score_q, cap, ext, inf = start_pipeline(
            simulate=simulate,
            camera_id=int(camera_id),
            encoder=encoder,
            drift_model=drift_model,
            cusum=CUSUMDetector(),
            video_queue=st.session_state.video_queue if simulate else None,
            webrtc_mode=not simulate,
            raw_features_queue=st.session_state.raw_features_queue
        )
        st.session_state.pipeline_stop = stop_event
        st.session_state.score_queue = score_q
        st.session_state.capture_thread = cap
        st.toast('Session started! 🚀')

    if stop_btn and st.session_state.session_active:
        st.session_state.session_active = False
        if st.session_state.pipeline_stop:
            st.session_state.pipeline_stop.set()
        st.toast('Session ended.')

    # --- Main Title ---
    st.markdown(
        '<div class="main-header">'
        '<h1 style="color:#ffffff; font-weight:800; font-size: 28px; margin: 0;">🧠 FATIGUE DRIFT DETECTOR</h1>'
        '<p style="color:#8a9ba8; font-size:14px; margin: 5px 0 0 0;">Personalized Driver Cognitive Fatigue Detection System</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Status notification banner
    status_banner_placeholder = st.empty()

    # --- Layout Grid ---
    row1_left, row1_right = st.columns([5, 7])

    with row1_left:
        video_container = st.container()
        with video_container:
            st.markdown('<h3 style="margin:0 0 12px 0; color:#fff; font-size:16px; font-weight:600;">📹 Live Video Feed</h3>', unsafe_allow_html=True)
            video_placeholder = st.empty()

    with row1_right:
        state_container = st.container()
        with state_container:
            st.markdown('<h3 style="margin:0 0 12px 0; color:#fff; font-size:16px; font-weight:600;">🧠 Current Driver State</h3>', unsafe_allow_html=True)
            gauge_placeholder = st.empty()
            metrics_placeholder = st.empty()

    st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)

    row2_left, row2_right = st.columns([7, 5])

    with row2_left:
        chart_container = st.container()
        with chart_container:
            st.markdown('<h3 style="margin:0 0 12px 0; color:#fff; font-size:16px; font-weight:600;">📈 Session Drift Trajectory</h3>', unsafe_allow_html=True)
            chart_placeholder = st.empty()

    with row2_right:
        contrib_container = st.container()
        with contrib_container:
            st.markdown('<h3 style="margin:0 0 12px 0; color:#fff; font-size:16px; font-weight:600;">📊 Top Feature Contributors</h3>', unsafe_allow_html=True)
            attribution_placeholder = st.empty()

    summary_placeholder = st.empty()

    # --- Define Placeholder Rendering Helper ---
    def render_placeholders(score, conf, alert, n_points):
        # 1. Update Gauge Indicator
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=score,
            number={'suffix': '%', 'font': {'size': 32, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#8a9ba8'},
                'bar': {'color': score_color(score), 'thickness': 0.75},
                'borderwidth': 1,
                'bordercolor': '#2c3e50',
                'steps': [
                    {'range': [0, 40],  'color': 'rgba(39, 174, 96, 0.15)'},
                    {'range': [40, 70], 'color': 'rgba(243, 156, 18, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.15)'},
                ],
                'threshold': {
                    'line': {'color': '#e74c3c', 'width': 3},
                    'value': 70,
                    'thickness': 0.85,
                },
            },
        ))
        fig_gauge.update_layout(
            height=130, margin=dict(l=10, r=10, t=5, b=5),
            paper_bgcolor='rgba(0,0,0,0)', font_color='#ecf0f1',
        )
        gauge_placeholder.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_chart_{n_points}_{time.time()}")

        # 2. Update Metrics Row
        status_str = score_label(score)
        color_hex = score_color(score)
        alert_str = alert_emoji(alert)
        
        html_metrics = (
            f'<div style="display: flex; justify-content: space-around; text-align: center; margin-top: 10px;">'
            f'<div style="flex: 1; border-right: 1px solid rgba(255,255,255,0.08);">'
            f'<div style="font-size: 11px; color: #8a9ba8; text-transform: uppercase; letter-spacing: 0.5px;">Status</div>'
            f'<div style="font-size: 15px; font-weight: 700; color: {color_hex}; margin-top: 4px;">{status_str}</div>'
            f'</div>'
            f'<div style="flex: 1; border-right: 1px solid rgba(255,255,255,0.08);">'
            f'<div style="font-size: 11px; color: #8a9ba8; text-transform: uppercase; letter-spacing: 0.5px;">Confidence</div>'
            f'<div style="font-size: 15px; font-weight: 700; color: #ffffff; margin-top: 4px;">{conf:.1f}%</div>'
            f'</div>'
            f'<div style="flex: 1; border-right: 1px solid rgba(255,255,255,0.08);">'
            f'<div style="font-size: 11px; color: #8a9ba8; text-transform: uppercase; letter-spacing: 0.5px;">CUSUM State</div>'
            f'<div style="font-size: 15px; font-weight: 700; color: #f39c12; margin-top: 4px;">{alert_str}</div>'
            f'</div>'
            f'<div style="flex: 1;">'
            f'<div style="font-size: 11px; color: #8a9ba8; text-transform: uppercase; letter-spacing: 0.5px;">Windows</div>'
            f'<div style="font-size: 15px; font-weight: 700; color: #ffffff; margin-top: 4px;">{n_points}</div>'
            f'</div>'
            f'</div>'
        )
        metrics_placeholder.markdown(html_metrics, unsafe_allow_html=True)

        # 3. Update Session Drift Chart
        if n_points > 1:
            fig_trend = go.Figure()
            times_min = [t / 60.0 for t in st.session_state.time_history]

            # Confidence band
            upper = [s + (100 - c) * 0.3 for s, c in
                     zip(st.session_state.score_history, st.session_state.confidence_history)]
            lower = [s - (100 - c) * 0.3 for s, c in
                     zip(st.session_state.score_history, st.session_state.confidence_history)]

            fig_trend.add_trace(go.Scatter(
                x=times_min + times_min[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(231,76,60,0.1)',
                line={'color': 'rgba(0,0,0,0)'},
                name='Confidence Band',
                showlegend=True,
            ))

            fig_trend.add_trace(go.Scatter(
                x=times_min, y=st.session_state.score_history,
                mode='lines+markers',
                name='Fatigue Score',
                line={'color': '#e74c3c', 'width': 3},
                marker={'size': 5, 'color': '#ffffff', 'line': {'width': 1.5, 'color': '#e74c3c'}},
            ))

            fig_trend.add_hline(y=70, line_dash='dash', line_color='#e74c3c',
                                annotation_text='High Fatigue Alert', annotation_position='top left')
            fig_trend.add_hline(y=40, line_dash='dot', line_color='#f39c12',
                                annotation_text='Mild Fatigue Alert', annotation_position='bottom left')

            # Mark CUSUM events cleanly with markers on the trace
            onset_x = []
            onset_y = []
            recovery_x = []
            recovery_y = []
            for i, alert_val in enumerate(st.session_state.alert_history):
                if alert_val == 'FATIGUE_ONSET':
                    onset_x.append(times_min[i])
                    onset_y.append(st.session_state.score_history[i])
                elif alert_val == 'RECOVERY':
                    recovery_x.append(times_min[i])
                    recovery_y.append(st.session_state.score_history[i])

            if onset_x:
                fig_trend.add_trace(go.Scatter(
                    x=onset_x, y=onset_y,
                    mode='markers',
                    name='Onset Event',
                    marker=dict(symbol='triangle-up', size=10, color='#e74c3c', line=dict(width=1, color='#ffffff')),
                    showlegend=True
                ))
            if recovery_x:
                fig_trend.add_trace(go.Scatter(
                    x=recovery_x, y=recovery_y,
                    mode='markers',
                    name='Recovery Event',
                    marker=dict(symbol='triangle-down', size=10, color='#27ae60', line=dict(width=1, color='#ffffff')),
                    showlegend=True
                ))

            fig_trend.update_layout(
                height=220,
                xaxis_title='Time (minutes)',
                yaxis_title='Fatigue Score',
                yaxis={'range': [0, 105], 'gridcolor': 'rgba(255,255,255,0.05)'},
                xaxis={'gridcolor': 'rgba(255,255,255,0.05)'},
                margin=dict(l=10, r=10, t=20, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#8a9ba8',
                legend=dict(orientation='h', yanchor='top', y=-0.3, xanchor='center', x=0.5),
            )

            chart_placeholder.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart_{n_points}_{time.time()}")
        else:
            chart_placeholder.info('Session drift trajectory will appear once enough windows are processed.')

        # 4. Update Feature Attributions (Display all 20 features in a scrollable view)
        if n_points > 0 and 'attributions_history' in st.session_state and st.session_state.attributions_history:
            latest_attr = st.session_state.attributions_history[-1]
            if not latest_attr:
                from src.explainability.attribution import FEATURE_NAMES
                latest_attr = [(f, 0.0) for f in FEATURE_NAMES]
            
            contrib_html = '<div style="max-height: 220px; overflow-y: auto; padding-right: 8px; margin-top: 5px;">'
            for name, pct in latest_attr:
                color = '#e74c3c' if pct > 15 else '#f39c12' if pct > 10 else '#27ae60'
                contrib_html += (
                    f'<div style="margin-bottom: 8px;">'
                    f'<div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px; color: #ecf0f1;">'
                    f'<span>{name}</span>'
                    f'<span style="font-weight: 600; color: {color};">{pct:.1f}%</span>'
                    f'</div>'
                    f'<div style="background: rgba(255,255,255,0.05); border-radius: 4px; height: 6px; width: 100%; overflow: hidden;">'
                    f'<div style="background: {color}; height: 100%; width: {pct}%; border-radius: 4px;"></div>'
                    f'</div>'
                    f'</div>'
                )
            contrib_html += '</div>'
            attribution_placeholder.markdown(contrib_html, unsafe_allow_html=True)
        else:
            attribution_placeholder.info('Feature attributions will appear during active sessions.')

    # --- Draw Initial (Inactive/Offline) Placeholders ---
    current_score = st.session_state.score_history[-1] if st.session_state.score_history else 0.0
    current_conf = st.session_state.confidence_history[-1] if st.session_state.confidence_history else 0.0
    current_alert = st.session_state.alert_history[-1] if st.session_state.alert_history else 'STABLE'
    n_points = len(st.session_state.score_history)

    # Set offline webcam image or render active WebRTC streamer
    if not st.session_state.session_active or simulate:
        offline_img = np.zeros((240, 320, 3), dtype=np.uint8) + 20
        import cv2
        cv2.putText(offline_img, "CAMERA OFFLINE", (45, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 110, 120), 2)
        cv2.putText(offline_img, "Click Start to activate feed", (35, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 90, 100), 1)
        video_placeholder.image(offline_img, channels="BGR", width="stretch")
    else:
        with video_placeholder.container():
            webrtc_ctx = webrtc_streamer(
                key="driver-fatigue-webrtc",
                video_processor_factory=WebRTCVideoProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                media_stream_constraints={"video": True, "audio": False},
            )
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.raw_features_queue = st.session_state.raw_features_queue

    # If inactive, render default/last values
    if not st.session_state.session_active:
        status_banner_placeholder.markdown(
            '<div class="connection-status" style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); color: #8a9ba8;">'
            '🔴 System Offline • Click Start in sidebar to begin monitoring'
            '</div>',
            unsafe_allow_html=True
        )
        render_placeholders(current_score, current_conf, current_alert, n_points)
        
        # Show session summary if we have stopped a session
        if n_points > 0:
            st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)
            summary_container = st.container(border=True)
            with summary_container:
                st.markdown('<h2 style="color:#ffffff; font-size: 20px; font-weight:700; margin-bottom: 15px;">📋 Session Summary Report</h2>', unsafe_allow_html=True)
                
                # Calculate average attributions across the entire session
                avg_attributions = {}
                for window_attrs in st.session_state.attributions_history:
                    for name, pct in window_attrs:
                        avg_attributions[name] = avg_attributions.get(name, 0.0) + pct
                
                if avg_attributions:
                    n_windows = len(st.session_state.attributions_history)
                    avg_attributions = {k: v / n_windows for k, v in avg_attributions.items()}
                    sorted_avg_attrs = sorted(avg_attributions.items(), key=lambda x: x[1], reverse=True)
                else:
                    sorted_avg_attrs = []

                col_left, col_right = st.columns([5, 5])
                with col_left:
                    st.markdown('<h3 style="color:#ffffff; font-size: 15px; font-weight:600; margin-bottom: 10px;">Session Performance</h3>', unsafe_allow_html=True)
                    scores = st.session_state.score_history
                    peak_fatigue = max(scores)
                    avg_fatigue = np.mean(scores)
                    onset_count = sum(1 for a in st.session_state.alert_history if a == 'FATIGUE_ONSET')
                    
                    if peak_fatigue >= 70.0 and onset_count > 0:
                        diag_html = (
                            '<div style="background: rgba(231, 76, 60, 0.15); border: 1px solid #e74c3c; '
                            'border-radius: 8px; padding: 12px; margin-bottom: 15px; color: #ff6b6b; '
                            'font-weight:600; font-size:14px;">🔴 CRITICAL: Severe fatigue detected during '
                            'the session. Driver rest is highly recommended.</div>'
                        )
                    elif peak_fatigue >= 40.0 or avg_fatigue >= 40.0:
                        diag_html = (
                            '<div style="background: rgba(243, 156, 18, 0.15); border: 1px solid #f39c12; '
                            'border-radius: 8px; padding: 12px; margin-bottom: 15px; color: #f39c12; '
                            'font-weight:600; font-size:14px;">🟡 WARNING: Mild to moderate driver fatigue '
                            'detected. Monitor driver closely.</div>'
                        )
                    else:
                        diag_html = (
                            '<div style="background: rgba(46, 204, 113, 0.15); border: 1px solid #2ecc71; '
                            'border-radius: 8px; padding: 12px; margin-bottom: 15px; color: #2ecc71; '
                            'font-weight:600; font-size:14px;">🟢 ALERT: Healthy driver alertness maintained. '
                            'No fatigue detected.</div>'
                        )
                    st.markdown(diag_html, unsafe_allow_html=True)
                    
                    m1, m2 = st.columns(2)
                    m1.metric('Peak Fatigue', f'{peak_fatigue:.1f}%')
                    m2.metric('Avg Fatigue', f'{avg_fatigue:.1f}%')
                    
                    m3, m4 = st.columns(2)
                    m3.metric('Min Fatigue', f'{min(scores):.1f}%')
                    recovery_count = sum(1 for a in st.session_state.alert_history if a == 'RECOVERY')
                    m4.metric('Onset / Recovery Events', f'{onset_count} / {recovery_count}')
                    
                    if st.session_state.time_history:
                        duration_min = st.session_state.time_history[-1] / 60.0
                        st.caption(f'Session duration: {duration_min:.1f} minutes • {n_points} windows processed')
                        
                with col_right:
                    st.markdown('<h3 style="color:#ffffff; font-size: 15px; font-weight:600; margin-bottom: 10px;">Session-Wide Contributors</h3>', unsafe_allow_html=True)
                    if sorted_avg_attrs:
                        summary_contrib_html = '<div style="max-height: 200px; overflow-y: auto; padding-right: 8px; margin-top: 5px;">'
                        for name, pct in sorted_avg_attrs[:5]:  # Display top 5
                            color = '#e74c3c' if pct > 15 else '#f39c12' if pct > 10 else '#27ae60'
                            summary_contrib_html += (
                                f'<div style="margin-bottom: 8px;">'
                                f'<div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px; color: #ecf0f1;">'
                                f'<span>{name}</span>'
                                f'<span style="font-weight: 600; color: {color};">{pct:.1f}%</span>'
                                f'</div>'
                                f'<div style="background: rgba(255,255,255,0.05); border-radius: 4px; height: 6px; width: 100%; overflow: hidden;">'
                                f'<div style="background: {color}; height: 100%; width: {pct}%; border-radius: 4px;"></div>'
                                f'</div>'
                                f'</div>'
                            )
                        summary_contrib_html += '</div>'
                        st.markdown(summary_contrib_html, unsafe_allow_html=True)
                    else:
                        st.info("No attribution data available.")

    # --- Active Real-Time Loop ---
    if st.session_state.session_active and st.session_state.score_queue:
        # Pre-render placeholders with current values immediately when active session starts
        render_placeholders(current_score, current_conf, current_alert, n_points)

        # Loop until session deactivated
        while st.session_state.session_active:
            # 1. Update Video Frame from queue (Simulation Mode only)
            if simulate and st.session_state.video_queue is not None:
                try:
                    frame = st.session_state.video_queue.get_nowait()
                    video_placeholder.image(frame, channels="BGR", width="stretch")
                except queue.Empty:
                    pass

            # 2. Check for new score metrics from model pipeline
            new_score_received = False
            try:
                while True:
                    data = st.session_state.score_queue.get_nowait()
                    elapsed = data['timestamp'] - st.session_state.session_start
                    st.session_state.score_history.append(data['score'])
                    st.session_state.time_history.append(elapsed)
                    st.session_state.confidence_history.append(data['confidence'])
                    st.session_state.alert_history.append(data['alert'])
                    st.session_state.attributions_history.append(data.get('attributions', []))
                    
                    current_score = data['score']
                    current_conf = data['confidence']
                    current_alert = data['alert']
                    n_points = len(st.session_state.score_history)
                    new_score_received = True
            except queue.Empty:
                pass

            # 3. Re-render Plotly/HTML widgets in place if score changed
            if new_score_received:
                render_placeholders(current_score, current_conf, current_alert, n_points)

            # 4. Check camera vs simulation status and update banner
            if st.session_state.get('capture_thread') is not None or not simulate:
                cap_thread = st.session_state.capture_thread
                if cap_thread is not None and cap_thread.simulate and not simulate:
                    status_banner_placeholder.markdown(
                        '<div class="connection-status" style="background: rgba(231, 76, 60, 0.1); border: 1px solid rgba(231, 76, 60, 0.3); color: #ff6b6b;">'
                        f'⚠️ CAMERA INITIALIZATION FAILED! Fell back to Simulation Mode (Camera ID: {camera_id})'
                        '</div>',
                        unsafe_allow_html=True
                    )
                elif simulate:
                    status_banner_placeholder.markdown(
                        '<div class="connection-status" style="background: rgba(241, 196, 15, 0.1); border: 1px solid rgba(241, 196, 15, 0.3); color: #f1c40f;">'
                        f'ℹ️ Running in Simulation Mode (Generating coherent driver sleepiness profiles)'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    if n_points < 6:
                        status_banner_placeholder.markdown(
                            f'<div class="connection-status" style="background: rgba(241, 196, 15, 0.1); border: 1px solid rgba(241, 196, 15, 0.3); color: #f1c40f;">'
                            f'🔄 CALIBRATING PERSONAL BASELINE... {n_points * 5}/30 seconds (Keep a neutral, alert face)'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        if not st.session_state.calibration_toast_shown:
                            st.toast("Personalized calibration complete! Monitoring active. 🎯")
                            st.session_state.calibration_toast_shown = True
                        status_banner_placeholder.markdown(
                            '<div class="connection-status" style="background: rgba(46, 204, 113, 0.1); border: 1px solid rgba(46, 204, 113, 0.3); color: #2ecc71;">'
                            f'🟢 Real-time WebRTC Webcam Active • Personalized Calibration Active'
                            '</div>',
                            unsafe_allow_html=True
                        )

            # Control polling/streaming frequency (~30 FPS)
            time.sleep(0.03)


if __name__ == '__main__':
    run_app()
