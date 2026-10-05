"""Stitch design tokens and shared components for Streamlit light and dark themes."""
import json
import streamlit as st


STITCH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Native Streamlit components use config.toml; these tokens theme custom HTML. */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--stitch-text);
}

[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    color: var(--stitch-text);
}

h1, h2, h3, h4, .stitch-headline {
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: var(--stitch-heading);
}

code, .stitch-mono { font-family: 'JetBrains Mono', monospace !important; }

section[data-testid="stSidebar"] {
    background-color: var(--stitch-app-bg) !important;
    border-right: 1px solid var(--stitch-border);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: var(--stitch-heading) !important; }

.stitch-sidebar-brand {
    padding: 4px 0 16px;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--stitch-border);
    display: flex;
    align-items: center;
    gap: 10px;
}
.stitch-brand-mark {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--stitch-brand) 0%, var(--stitch-info) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 800;
    font-size: 18px;
    font-family: 'Hanken Grotesk', sans-serif;
}
.stitch-brand-name { font-family: 'Hanken Grotesk', sans-serif; font-size: 17px; font-weight: 700; color: var(--stitch-heading); line-height: 1.1; }
.stitch-brand-caption { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--stitch-muted); }
.stitch-sidebar-footer { margin-top: 24px; padding: 14px; background: var(--stitch-inset); border-radius: 8px; border: 1px solid var(--stitch-border); color: var(--stitch-text); }
.stitch-sidebar-status { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; color: var(--stitch-brand-strong); font-size: 11px; font-weight: 600; }
.stitch-sidebar-active { background: var(--stitch-success-bg); color: var(--stitch-success); border: 1px solid var(--stitch-success-border); font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 9999px; }
.stitch-sidebar-detail { font-size: 11px; color: var(--stitch-secondary); margin-bottom: 4px; }
.stitch-sidebar-muted { font-size: 10px; color: var(--stitch-muted); margin-bottom: 6px; }
.stitch-sidebar-version { display: flex; justify-content: space-between; border-top: 1px solid var(--stitch-border); padding-top: 6px; font-size: 10px; color: var(--stitch-muted); }
.stitch-sidebar-version strong { color: var(--stitch-text); }
.stitch-page-header { margin-bottom: 24px; }
.stitch-page-tag { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--stitch-info); }
.stitch-tag-dot { width: 8px; height: 8px; border-radius: 999px; background: var(--stitch-info); }
.stitch-page-header h1 { font-size: 28px; line-height: 1.2; margin: 0 0 6px; }
.stitch-page-header p { font-size: 14px; color: var(--stitch-secondary); margin: 0; max-width: 800px; line-height: 1.5; }
.stitch-department-tag { color: var(--stitch-muted); font-size: 11px; }
.stitch-confidence-row { margin-bottom: 12px; color: var(--stitch-text); }
.stitch-confidence-label { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: var(--stitch-text); }
.stitch-confidence-track { width: 100%; height: 7px; background: var(--stitch-inset); border-radius: 9999px; overflow: hidden; }
.stitch-confidence-track > div { height: 100%; border-radius: 9999px; }

.stitch-kpi-card, .stitch-strip, .stitch-result-card {
    color: var(--stitch-text);
    background: var(--stitch-surface);
    border: 1px solid var(--stitch-border);
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}

.stitch-kpi-card {
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stitch-kpi-card:hover, .stitch-result-card:hover, .stitch-strip:hover {
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.16);
}

.stitch-kpi-title {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--stitch-muted);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.stitch-kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: var(--stitch-text);
    line-height: 1.15;
    margin-bottom: 6px;
}

.stitch-kpi-sub { font-size: 12px; color: var(--stitch-muted); display: flex; justify-content: space-between; }

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

.badge-emerald { background: var(--stitch-success-bg); color: var(--stitch-success); border: 1px solid var(--stitch-success-border); }
.badge-amber { background: var(--stitch-warning-bg); color: var(--stitch-warning); border: 1px solid var(--stitch-warning-border); }
.badge-rose { background: var(--stitch-danger-bg); color: var(--stitch-danger); border: 1px solid var(--stitch-danger-border); }
.badge-blue { background: var(--stitch-info-bg); color: var(--stitch-info); border: 1px solid var(--stitch-info-border); }
.badge-gray { background: var(--stitch-neutral-bg); color: var(--stitch-neutral); border: 1px solid var(--stitch-neutral-border); }

.stitch-strip { padding: 16px 20px; margin-bottom: 24px; }
.stitch-strip-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--stitch-border);
    margin-bottom: 14px;
    font-size: 12px;
    font-weight: 600;
    color: var(--stitch-secondary);
}

.stitch-result-card { padding: 24px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12); }
.stitch-banner { padding: 12px 16px; border-radius: 8px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; }
.banner-high { background: var(--stitch-success-bg); color: var(--stitch-success); border: 1px solid var(--stitch-success-border); }
.banner-medium { background: var(--stitch-warning-bg); color: var(--stitch-warning); border: 1px solid var(--stitch-warning-border); }
.banner-low { background: var(--stitch-danger-bg); color: var(--stitch-danger); border: 1px solid var(--stitch-danger-border); }

.stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div {
    color: var(--stitch-text) !important;
    background-color: var(--stitch-surface) !important;
    border: 1px solid var(--stitch-border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

.stTextArea textarea::placeholder, .stTextInput input::placeholder { color: var(--stitch-muted) !important; opacity: 0.85; }
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus,
.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: var(--stitch-brand) !important;
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--stitch-brand) 30%, transparent) !important;
}

.stButton > button, [data-testid="stFormSubmitButton"] button, .stDownloadButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    transition: filter 0.15s ease, box-shadow 0.15s ease !important;
}

.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
    background-color: var(--stitch-brand-fill) !important;
    color: #ffffff !important;
    border: 1px solid var(--stitch-brand-fill) !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { filter: brightness(1.12); }
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 3px solid var(--stitch-brand) !important;
    outline-offset: 2px !important;
}

div[data-testid="stDataFrame"], div[data-testid="stTable"] { border: 1px solid var(--stitch-border); border-radius: 8px; overflow: hidden; }
div[data-testid="stDataFrame"] *, div[data-testid="stTable"] * { color-scheme: inherit; }
[data-testid="stAlert"] { border-radius: 8px; }
</style>
"""


_STITCH_TOKENS = {
    "light": {
        "color-scheme": "light", "app-bg": "#f8f9ff", "surface": "#ffffff", "inset": "#f8f9ff",
        "text": "#0b1c30", "heading": "#00236f", "secondary": "#444651", "muted": "#64748b",
        "border": "#dce3ee", "brand": "#1e3a8a", "brand-fill": "#1e3a8a", "brand-strong": "#1e3a8a",
        "success": "#047857", "success-bg": "#ecfdf5", "success-border": "#059669",
        "warning": "#b45309", "warning-bg": "#fffbeb", "warning-border": "#d97706",
        "danger": "#be123c", "danger-bg": "#fff1f2", "danger-border": "#e11d48",
        "info": "#1d4ed8", "info-bg": "#eff6ff", "info-border": "#2563eb",
        "neutral": "#475569", "neutral-bg": "#f1f5f9", "neutral-border": "#cbd5e1",
    },
    "dark": {
        "color-scheme": "dark", "app-bg": "#0d1422", "surface": "#182235", "inset": "#121c2d",
        "text": "#e6edf7", "heading": "#d7e4ff", "secondary": "#c3cede", "muted": "#a9b7cb",
        "border": "#35445a", "brand": "#76a2ff", "brand-fill": "#2458b8", "brand-strong": "#a8c4ff",
        "success": "#a8f0ca", "success-bg": "#163829", "success-border": "#48c78e",
        "warning": "#ffd28a", "warning-bg": "#3b2c16", "warning-border": "#f0ad4e",
        "danger": "#ffb2c0", "danger-bg": "#3d202b", "danger-border": "#ff6b86",
        "info": "#b7d0ff", "info-bg": "#1c3152", "info-border": "#75a7ff",
        "neutral": "#c6d0df", "neutral-bg": "#263247", "neutral-border": "#53627a",
    },
}


def _theme_token_css(theme_type):
    """Build custom HTML color tokens from the active Streamlit theme type."""
    tokens = _STITCH_TOKENS["dark" if theme_type == "dark" else "light"]
    css_vars = "\n".join(f"--stitch-{name}: {value};" for name, value in tokens.items() if name != "color-scheme")
    sidebar = dict(tokens)
    if theme_type == "dark":
        sidebar.update({"app-bg": "#0a1020", "surface": "#121c2d", "inset": "#0d1422", "border": "#2c3a50"})
    else:
        sidebar.update({"app-bg": "#ffffff", "surface": "#f8f9ff", "inset": "#f8f9ff"})
    sidebar_vars = "\n".join(f"--stitch-{name}: {value};" for name, value in sidebar.items() if name != "color-scheme")
    return f"""
    <style>
    :root {{ color-scheme: {tokens['color-scheme']}; {css_vars} }}
    section[data-testid="stSidebar"] {{ {sidebar_vars} }}
    </style>
    """


def apply_stitch_styles():
    """Apply Streamlit-native colors to custom Stitch HTML for the active theme."""
    try:
        theme_type = st.context.theme.type
    except (AttributeError, RuntimeError, KeyError):
        theme_type = "light"
    st.markdown(_theme_token_css(theme_type) + STITCH_CSS, unsafe_allow_html=True)
    # Streamlit switches native themes in the browser without rerunning Python.
    # A tiny same-origin component watches that native surface and requests one
    # rerun on a real mode change so custom HTML and Matplotlib use fresh tokens.
    st.components.v1.html(_theme_watch_script(theme_type), height=0, width=0)


def _theme_watch_script(initial_theme):
    """Keep custom HTML and rasterized plots aligned with client-side theme changes.

    Streamlit applies its Settings-menu theme client-side. Its context theme may be
    stale during that transition, and raw Markdown HTML does not receive the
    custom-component CSS variables. This trusted local script only reads the
    rendered app background; it never reads or transmits app data.
    """
    theme_json = json.dumps(_STITCH_TOKENS)
    return f"""
    <script>
    (() => {{
      const doc = window.parent.document;
      const root = doc.documentElement;
      const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
      const themes = {theme_json};
      const initialTheme = {json.dumps(initial_theme)};
      const storageKey = 'supportiq-stitch-theme';
      let scheduled = false;
      let appliedMode = null;

      function luminance(color) {{
        const rgb = color.match(/[\\d.]+/g);
        if (!rgb || rgb.length < 3) return 1;
        const channels = rgb.slice(0, 3).map(value => {{
          const normalized = Number(value) / 255;
          return normalized <= 0.04045 ? normalized / 12.92 : Math.pow((normalized + 0.055) / 1.055, 2.4);
        }});
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
      }}

      function applyTheme() {{
        scheduled = false;
        const app = doc.querySelector('[data-testid="stApp"]');
        if (!app) return;
        const background = getComputedStyle(app).backgroundColor;
        const mode = luminance(background) < 0.5 ? 'dark' : 'light';
        if (mode === appliedMode) return;
        appliedMode = mode;
        const tokens = themes[mode];
        root.style.colorScheme = tokens['color-scheme'];
        Object.entries(tokens).filter(([name]) => name !== 'color-scheme')
          .forEach(([name, value]) => root.style.setProperty('--stitch-' + name, value));
        if (sidebar) {{
          Object.entries(tokens).filter(([name]) => name !== 'color-scheme')
            .forEach(([name, value]) => sidebar.style.setProperty('--stitch-' + name, value));
        }}

        const previous = window.parent.localStorage.getItem(storageKey);
        window.parent.localStorage.setItem(storageKey, mode);
        if ((previous && previous !== mode) || (!previous && mode !== initialTheme)) {{
          window.parent.location.reload();
        }}
      }}

      function scheduleApply() {{
        if (!scheduled) {{ scheduled = true; requestAnimationFrame(applyTheme); }}
      }}
      const observer = new MutationObserver(scheduleApply);
      observer.observe(doc.head, {{ childList: true, subtree: true }});
      observer.observe(doc.body, {{ childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style'] }});
      applyTheme();
      window.setInterval(applyTheme, 1000);
    }})();
    </script>
    """


def get_plot_colors():
    """Return readable Matplotlib colors for the active Streamlit theme."""
    try:
        dark = st.context.theme.type == "dark"
    except (AttributeError, RuntimeError, KeyError):
        dark = False

    if dark:
        return {
            "background": "#0d1422", "surface": "#182235", "text": "#e6edf7",
            "heading": "#d7e4ff", "muted": "#a9b7cb", "border": "#35445a",
            "grid": "#43536a", "primary": "#76a2ff", "secondary": "#4f8de8",
            "tertiary": "#a8b6ca", "success": "#48c78e", "warning": "#f0ad4e",
            "danger": "#ff6b86",
        }
    return {
        "background": "#f8f9ff", "surface": "#ffffff", "text": "#0b1c30",
        "heading": "#00236f", "muted": "#64748b", "border": "#e2e8f0",
        "grid": "#cbd5e1", "primary": "#1e3a8a", "secondary": "#2563eb",
        "tertiary": "#64748b", "success": "#059669", "warning": "#d97706",
        "danger": "#e11d48",
    }


def style_plot_axes(fig, ax):
    """Apply active-theme surfaces, text, and subtle borders to a Matplotlib plot."""
    colors = get_plot_colors()
    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor(colors["surface"])
    ax.tick_params(colors=colors["muted"])
    for spine in ax.spines.values():
        spine.set_color(colors["border"])
    return colors


def render_sidebar_branding():
    """Render SupportIQ branded sidebar header."""
    st.sidebar.markdown(
        """
        <div class="stitch-sidebar-brand">
            <div class="stitch-brand-mark">IQ</div>
            <div>
                <div class="stitch-brand-name">SupportIQ</div>
                <div class="stitch-brand-caption">ML Operations Console</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer():
    """Render system telemetry card at the bottom of the sidebar."""
    st.sidebar.markdown(
        """
        <div class="stitch-sidebar-footer">
            <div class="stitch-sidebar-status">
                <span>Model: Logistic Regression</span>
                <span class="stitch-sidebar-active">Active</span>
            </div>
            <div class="stitch-sidebar-detail">Dataset: <strong>BANKING77</strong></div>
            <div class="stitch-sidebar-muted">10,003 train · 3,080 test (77 intents)</div>
            <div class="stitch-sidebar-version"><span>System Version</span><strong>v1.1-ui</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, tag: str = "PRODUCTION PIPELINE"):
    """Render standard Stitch view header with breadcrumb and telemetry tag."""
    st.markdown(
        f"""
        <div class="stitch-page-header">
            <div class="stitch-page-tag"><span class="stitch-tag-dot"></span><span>{tag}</span></div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
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
            <div class="stitch-kpi-title"><span>{title}</span>{badge_html}</div>
            <div class="stitch-kpi-value">{value}</div>
        </div>
        <div class="stitch-kpi-sub"><span>{subtext}</span></div>
    </div>
    """


def render_confidence_bar(intent: str, confidence: float, department: str = ""):
    """Render a styled probability bar with theme-aware surfaces and semantic color."""
    pct = round(confidence * 100, 1)
    if confidence >= 0.85:
        bar_color, badge_class = "var(--stitch-success)", "badge-emerald"
    elif confidence >= 0.60:
        bar_color, badge_class = "var(--stitch-warning)", "badge-amber"
    else:
        bar_color, badge_class = "var(--stitch-danger)", "badge-rose"

    dept_tag = f'<span class="stitch-department-tag">→ {department}</span>' if department else ""
    st.markdown(
        f"""
        <div class="stitch-confidence-row">
            <div class="stitch-confidence-label"><span>{intent} {dept_tag}</span>
                <span class="stitch-badge {badge_class}">{pct}%</span>
            </div>
            <div class="stitch-confidence-track"><div style="width: {pct}%; background: {bar_color};"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
