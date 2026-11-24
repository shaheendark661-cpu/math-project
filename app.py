import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
import streamlit as st

# --- Page & Theme Configuration ---
st.set_page_config(
    page_title="Structural Numerical Integration",
    page_icon="📐",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }
        .header-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            padding: 1.5rem 2rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.12);
        }
        .header-card h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
        }
        .header-card p {
            margin: 0.35rem 0 0 0;
            color: #cbd5e1;
            font-size: 0.95rem;
        }
        .metric-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.2rem;
        }
        .metric-sub {
            font-size: 0.85rem;
            color: #0369a1;
            margin-top: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-card">
        <h1>📐 Irregular Structural Geometry — Numerical Integration</h1>
        <p>Compare Trapezoidal &amp; Simpson's rules against exact analytical values with error metrics, curve plot, and a live QR code.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def make_qr_image(url: str, scale: int = 10, border: int = 40) -> Image.Image:
    """Render a real scannable QR code PIL image from a URL."""
    widget = qr.QrCodeWidget(url.strip(), barLevel="M")
    bounds = widget.getBounds()
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]

    img_w = int(round(w * scale)) + 2 * border
    img_h = int(round(h * scale)) + 2 * border
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    group = widget.draw()

    def draw_rects(node):
        if hasattr(node, "contents"):
            for child in node.contents:
                draw_rects(child)
        elif hasattr(node, "x") and hasattr(node, "y") and hasattr(node, "width") and hasattr(node, "height"):
            col = getattr(node, "fillColor", None)
            if col is not None and (col.red, col.green, col.blue) == (0, 0, 0):
                x0 = border + node.x * scale
                y0 = border + (h - (node.y + node.height)) * scale
                x1 = border + (node.x + node.width) * scale
                y1 = border + (h - node.y) * scale
                draw.rectangle([x0, y0, x1, y1], fill="black")

    draw_rects(group)
    return img


# --- Numerical Integration Functions ---
def trapezoidal_rule(func, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return float((h / 2.0) * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1]))


def simpsons_rule(func, a: float, b: float, n: int) -> float:
    effective_n = n if n % 2 == 0 else n + 1
    h = (b - a) / effective_n
    x = np.linspace(a, b, effective_n + 1)
    y = func(x)
    odd_sum = np.sum(y[1:-1:2])
    even_sum = np.sum(y[2:-2:2])
    return float((h / 3.0) * (y[0] + 4.0 * odd_sum + 2.0 * even_sum + y[-1]))


def pct_error(approx: float, exact: float) -> float:
    if exact == 0:
        return 0.0
    return float(abs(approx - exact) / abs(exact) * 100.0)


# --- Sidebar Controls ---
st.sidebar.header("Geometry & Solver Settings")

preset = st.sidebar.selectbox(
    "Select Engineering Problem",
    [
        "Parabolic Arch Cross-Section (Area)",
        "Curved Haunch Beam Profile (Area)",
        "Horizontal Cylindrical Tank (Volume)",
    ],
)

n_intervals = st.sidebar.slider(
    "Number of Sub-intervals (n)",
    min_value=2,
    max_value=200,
    value=10,
    step=2,
    help="Simpson's rule uses an even interval count.",
)

method_overlay = st.sidebar.radio(
    "Highlight Sub-interval Partition in Plot",
    ["Simpson's Rule (Parabolic)", "Trapezoidal Rule (Linear)"],
)

st.sidebar.markdown("---")
st.sidebar.header("📱 Phone QR Code Link")
app_url = st.sidebar.text_input(
    "Paste your live https:// app URL",
    value="https://share.streamlit.io",
    help="Once hosted free on Streamlit Community Cloud, paste your app's public https:// link here.",
)

# --- Problem Definitions ---
if preset == "Parabolic Arch Cross-Section (Area)":
    st.sidebar.subheader("Arch Parameters (meters)")
    span = st.sidebar.number_input("Span L (m)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
    rise = st.sidebar.number_input("Crown Height H (m)", min_value=0.5, max_value=50.0, value=4.0, step=0.5)

    a, b = 0.0, span
    func = lambda x: (4.0 * rise * x * (span - x)) / (span ** 2)
    exact_val = (2.0 / 3.0) * span * rise
    unit = "m²"
    y_label = "Height y(x) [m]"
    x_label = "Position along Span x [m]"
    formula_desc = "y(x) = 4H·x(L - x) / L²"

elif preset == "Curved Haunch Beam Profile (Area)":
    st.sidebar.subheader("Haunch Beam Parameters (meters)")
    length = st.sidebar.number_input("Beam Length L (m)", min_value=1.0, max_value=100.0, value=8.0, step=0.5)
    d_min = st.sidebar.number_input("Midspan Depth d₀ (m)", min_value=0.1, max_value=10.0, value=0.6, step=0.1)
    d_max = st.sidebar.number_input("Support Depth d₁ (m)", min_value=0.2, max_value=15.0, value=1.4, step=0.1)

    a, b = -length / 2.0, length / 2.0
    func = lambda x: d_min + (d_max - d_min) * ((2.0 * x / length) ** 2)
    exact_val = length * (d_min + (d_max - d_min) / 3.0)
    unit = "m²"
    y_label = "Beam Depth d(x) [m]"
    x_label = "Distance from Midspan x [m]"
    formula_desc = "d(x) = d₀ + (d₁ - d₀)(2x / L)²"

else:
    st.sidebar.subheader("Tank Parameters (meters)")
    radius = st.sidebar.number_input("Tank Radius R (m)", min_value=0.5, max_value=20.0, value=2.0, step=0.25)
    tank_len = st.sidebar.number_input("Tank Length L (m)", min_value=1.0, max_value=100.0, value=6.0, step=0.5)
    fill_h = st.sidebar.slider(
        "Liquid Depth h (m)",
        min_value=0.1,
        max_value=float(2.0 * radius),
        value=float(1.3 * radius),
        step=0.05,
    )

    a, b = 0.0, tank_len
    cos_arg = np.clip((radius - fill_h) / radius, -1.0, 1.0)
    radicand = max(0.0, 2.0 * radius * fill_h - fill_h ** 2)
    cross_area = (radius ** 2) * np.arccos(cos_arg) - (radius - fill_h) * np.sqrt(radicand)

    func = lambda z: np.full_like(z, cross_area, dtype=float)
    exact_val = cross_area * tank_len
    unit = "m³"
    y_label = "Cross-Sectional Liquid Area A(z) [m²]"
    x_label = "Axial Position along Tank z [m]"
    formula_desc = "A(z) = R²·arccos((R - h)/R) - (R - h)√(2Rh - h²)"

# --- Compute Numerical Results ---
trap_val = trapezoidal_rule(func, a, b, n_intervals)
simp_val = simpsons_rule(func, a, b, n_intervals)
trap_err = pct_error(trap_val, exact_val)
simp_err = pct_error(simp_val, exact_val)
eff_n = n_intervals if n_intervals % 2 == 0 else n_intervals + 1

# --- Top Summary Cards ---
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Analytical (Exact)</div>
            <div class="metric-value">{exact_val:.6f} {unit}</div>
            <div class="metric-sub">{formula_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Trapezoidal Rule (n = {n_intervals})</div>
            <div class="metric-value">{trap_val:.6f} {unit}</div>
            <div class="metric-sub">Error: {trap_err:.4f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Simpson's Rule (n = {eff_n})</div>
            <div class="metric-value">{simp_val:.6f} {unit}</div>
            <div class="metric-sub">Error: {simp_err:.4f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### ")

# --- Plot & Comparison Table ---
left_col, right_col = st.columns([1.45, 1.0])

with left_col:
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    x_dense = np.linspace(a, b, 400)
    y_dense = func(x_dense)

    ax.plot(x_dense, y_dense, color="#0f172a", linewidth=2.2, label="Exact Profile")
    ax.fill_between(x_dense, 0, y_dense, color="#93c5fd", alpha=0.35, label="Integrated Region")

    x_nodes = np.linspace(a, b, eff_n + 1)
    y_nodes = func(x_nodes)

    if method_overlay == "Trapezoidal Rule (Linear)":
        ax.plot(x_nodes, y_nodes, "o--", color="#dc2626", linewidth=1.3, markersize=4, label="Trapezoidal Chords")
        for i in range(len(x_nodes) - 1):
            ax.plot([x_nodes[i], x_nodes[i]], [0, y_nodes[i]], color="#dc2626", linewidth=0.5, alpha=0.45)
    else:
        for i in range(0, len(x_nodes) - 1, 2):
            xs_pair = x_nodes[i : i + 3]
            ys_pair = y_nodes[i : i + 3]
            if len(xs_pair) == 3:
                poly = np.polyfit(xs_pair, ys_pair, 2)
                xp = np.linspace(xs_pair[0], xs_pair[2], 30)
                yp = np.polyval(poly, xp)
                ax.plot(
                    xp,
                    yp,
                    "--",
                    color="#059669",
                    linewidth=1.5,
                    label="Simpson Parabolas" if i == 0 else "",
                )
        ax.scatter(x_nodes, y_nodes, color="#059669", s=18, zorder=4)

    ax.set_title(f"{preset}", fontsize=11.5, fontweight="bold", color="#0f172a", pad=10)
    ax.set_xlabel(x_label, fontsize=9.5, color="#334155")
    ax.set_ylabel(y_label, fontsize=9.5, color="#334155")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8.5, frameon=True)
    plt.tight_layout()
    st.pyplot(fig)

with right_col:
    st.markdown("#### Results Comparison")
    st.dataframe(
        {
            "Method": ["Analytical (Exact)", "Trapezoidal Rule", "Simpson's Rule"],
            f"Result ({unit})": [
                round(exact_val, 6),
                round(trap_val, 6),
                round(simp_val, 6),
            ],
            "Abs. Diff": [
                0.0,
                round(abs(trap_val - exact_val), 6),
                round(abs(simp_val - exact_val), 6),
            ],
            "Error (%)": [
                0.0,
                round(trap_err, 5),
                round(simp_err, 5),
            ],
        },
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Scan to Open Online")
    if app_url.strip().startswith(("http://", "https://")):
        qr_img = make_qr_image(app_url)
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        st.image(qr_img, width=180, caption=app_url.strip())
        st.download_button(
            "Download QR Code (PNG)",
            data=buf.getvalue(),
            file_name="structural_app_qr.png",
            mime="image/png",
        )
    else:
        st.warning("Enter a valid http:// or https:// URL in the sidebar to generate a scannable QR code.")
