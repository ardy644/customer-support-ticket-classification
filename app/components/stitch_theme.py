"""Stitch Design System theme and shared components for Streamlit.

Implements the Fintech Intelligence design specification:
- Fonts: Hanken Grotesk, Inter, JetBrains Mono
- Color Palette: Deep Navy, Cobalt, Slate, Emerald, Amber, Rose
- Components: Metric Cards, Status Badges, Probability Bars, Navigation
"""
import streamlit as st

# Stitch Design System Color Palette
COLORS = {
    'primary': '#00236f',
    'primary_container': '#1e3a8a',
    'on_primary': '#ffffff',
    'secondary': '#0051d5',
    'secondary_container': '#316bf3',
    'surface': '#f8f9ff',
    'surface_low': '#eff4ff',
    'surface_high': '#dce9ff',
    'surface_card': '#ffffff',
    'on_surface': '#0b1c30',
    'on_surface_variant': '#444651',
    'outline': '#757682',
    'outline_light': '#e2e8f0',
    'emerald': '#059669',
    'emerald_bg': '#ecfdf5',
    'amber': '#d97706',
    'amber_bg': '#fffbeb',
    'rose': '#e11d48',
    'rose_bg': '#fff1f2',
}

STITCH_CSS = """
<style>
/* Import Stitch typography */
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Base Styles */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0b1c30;
}

h1, h2, h3, h4, .stitch-headline {
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: #00236f;
}

code, .stitch-mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* App Background */
.stApp {
    background-color: #f8f9ff;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00236f !important;
}

/* KPI Card */
.stitch-kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stitch-kpi-card:hover {
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}

.stitch-kpi-title {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.stitch-kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #0b1c30;
    line-height: 1.15;
    margin-bottom: 6px;
}

.stitch-kpi-sub {
    font-size: 12px;
    color: #64748b;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Badges */
.stitch-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}

.badge-emerald {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}

.badge-amber {
    background: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
}

.badge-rose {
    background: #fff1f2;
    color: #e11d48;
    border: 1px solid #fecdd3;
}

.badge-blue {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

.badge-gray {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
}

/* Diagnostic Health Strip */
.stitch-strip {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    margin-bottom: 24px;
}

.stitch-strip-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 14px;
    font-size: 12px;
    font-weight: 600;
    color: #475569;
}

/* Prediction Result Card */
.stitch-result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}

.stitch-banner {
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.banner-high {
    background: #004a32;
    color: #ffffff;
}

.banner-medium {
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
}

.banner-low {
    background: #fff1f2;
    color: #9f1239;
    border: 1px solid #fecdd3;
}

/* Clean Input Styling */
.stTextArea textarea {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

.stTextArea textarea:focus {
    border-color: #1e3a8a !important;
    box-shadow: 0 0 0 2px rgba(30, 58, 138, 0.15) !important;
}

/* Primary Button */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="primary"] {
    background-color: #1e3a8a !important;
    color: #ffffff !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1e40af !important;
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.25) !important;
}

/* Dataframe & Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""


def apply_stitch_styles():
    """Inject Stitch Design System CSS styles into the current Streamlit page."""
    st.markdown(STITCH_CSS, unsafe_allow_html=True)


def render_sidebar_branding():
    """Render SupportIQ branded sidebar header and system telemetry footer."""
    st.sidebar.markdown(
        """
        <div style="padding: 4px 0 16px 0; border-bottom: 1px solid #f1f5f9; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 18px; font-family: 'Hanken Grotesk', sans-serif;">
                    IQ
                </div>
                <div>
                    <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 17px; font-weight: 700; color: #00236f; line-height: 1.1;">
                        SupportIQ
                    </div>
                    <div style="font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b;">
                        ML Operations Console
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer():
    """Render system telemetry card at the bottom of the sidebar."""
    st.sidebar.markdown(
        """
        <div style="margin-top: 24px; padding: 14px; background: #eff4ff; border-radius: 8px; border: 1px solid #dce9ff;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 11px; font-weight: 600; color: #1e3a8a;">Model: Logistic Regression</span>
                <span style="background: #85f8c4; color: #002114; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 9999px;">Active</span>
            </div>
            <div style="font-size: 11px; color: #444651; margin-bottom: 4px;">
                Dataset: <strong>BANKING77</strong>
            </div>
            <div style="font-size: 10px; color: #757682; margin-bottom: 6px;">
                10,003 train · 3,080 test (77 intents)
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 10px; color: #64748b; padding-top: 6px; border-top: 1px solid #dce9ff;">
                <span>System Version</span>
                <span style="font-weight: 600; color: #0b1c30;">v1.1-ui</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, tag: str = "PRODUCTION PIPELINE"):
    """Render standard Stitch view header with breadcrumb and telemetry tag."""
    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #2563eb; margin-bottom: 4px;">
                <span style="width: 8px; height: 8px; border-radius: 9999px; background: #2563eb; display: inline-block;"></span>
                <span>{tag}</span>
            </div>
            <h1 style="font-size: 28px; line-height: 1.2; margin: 0 0 6px 0;">{title}</h1>
            <p style="font-size: 14px; color: #444651; margin: 0; max-width: 800px; line-height: 1.5;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, subtext: str, badge_text: str = "", badge_type: str = "emerald"):
    """HTML snippet for a single Stitch KPI Card."""
    badge_html = f'<span class="stitch-badge badge-{badge_type}">{badge_text}</span>' if badge_text else ""
    return f"""
    <div class="stitch-kpi-card">
        <div>
            <div class="stitch-kpi-title">
                <span>{title}</span>
                {badge_html}
            </div>
            <div class="stitch-kpi-value">{value}</div>
        </div>
        <div class="stitch-kpi-sub">
            <span>{subtext}</span>
        </div>
    </div>
    """


def render_confidence_bar(intent: str, confidence: float, department: str = ""):
    """Render a styled probability progress bar according to Stitch confidence tiers."""
    pct = round(confidence * 100, 1)
    if confidence >= 0.85:
        bar_color = "#059669"
        badge_class = "badge-emerald"
    elif confidence >= 0.60:
        bar_color = "#d97706"
        badge_class = "badge-amber"
    else:
        bar_color = "#e11d48"
        badge_class = "badge-rose"

    dept_tag = f'<span style="color: #64748b; font-size: 11px;">→ {department}</span>' if department else ""

    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: #0b1c30;">
                    {intent} {dept_tag}
                </span>
                <span class="stitch-badge {badge_class}" style="font-family: 'JetBrains Mono', monospace;">
                    {pct}%
                </span>
            </div>
            <div style="width: 100%; height: 7px; background: #eff4ff; border-radius: 9999px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; background: {bar_color}; border-radius: 9999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
