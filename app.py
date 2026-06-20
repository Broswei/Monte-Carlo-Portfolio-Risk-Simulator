"""Monte Carlo Portfolio Risk Simulator - Streamlit UI"""

from __future__ import annotations
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
from risk import (
    run_simulation,
    simulate_paths,
    histogram,
    estimate_from_prices,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    min_variance_weights,
    max_sharpe_weights,
    efficient_frontier,
    portfolio_stats,
    TRADING_DAYS,
)

INK = "#E9E3D5"
INK_DIM = "#9DAFB4"
INK_FAINT = "#5F7177"
AMBER = "#E0A33E"
LOSS = "#D85A4A"
GAIN = "#4FB89E"
COOL = "#3E7DA6"
PANEL = "#13242B"
GRID = "#1E323A"

DEFAULT_ASSETS = pd.DataFrame(
    {
        "Asset": ["US Equity", "Intl Equity", "Bonds", "Gold", "REIT"],
        "Return %/yr": [8.0, 6.5, 3.0, 4.0, 7.0],
        "Vol %/yr": [16.0, 18.0, 6.0, 15.0, 20.0],
        "Weight %": [40.0, 20.0, 25.0, 10.0, 5.0],
    }
)
DEFAULT_CORR = np.array(
    [
        [1.00, 0.78, -0.10, 0.05, 0.62],
        [0.78, 1.00, -0.05, 0.10, 0.55],
        [-0.10, -0.05, 1.00, 0.20, 0.00],
        [0.05, 0.10, 0.20, 1.00, 0.08],
        [0.62, 0.55, 0.00, 0.08, 1.00],
    ]
)

st.set_page_config(
    page_title="Monte Carlo Portfolio Risk Simulator",
    page_icon="📉",
    layout="wide",
)
 
st.markdown(
    """
    <style>
      .stMetric { background:#0F1F25; border:1px solid #23393F;
                  border-radius:6px; padding:12px 14px; }
      div[data-testid="stMetricValue"] { font-family: ui-monospace, monospace; }
      .eyebrow { font-family: ui-monospace, monospace; letter-spacing:.32em;
                 color:#E0A33E; text-transform:uppercase; font-size:12px; }
      .muted { color:#5F7177; font-family: ui-monospace, monospace; font-size:12px; }
      h1 { letter-spacing:-.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "assets" not in st.session_state:
    st.session_state.assets = DEFAULT_ASSETS.copy()
    st.session_state.corr = DEFAULT_CORR.copy()
    st.session_state.result = None
    st.session_state.warning = None
    st.session_state.perf = None
    st.session_state.steps = None
 
 
def sync_corr_size(n: int, names: list[str]) -> None:
    """Resize the correlation matrix to match the asset list, keeping overlap."""
    cur = st.session_state.corr
    if cur.shape[0] == n:
        return
    new = np.eye(n)
    k = min(n, cur.shape[0])
    new[:k, :k] = cur[:k, :k]
    np.fill_diagonal(new, 1.0)
    st.session_state.corr = new

st.markdown('<div class="eyebrow">Monte Carlo</div>', unsafe_allow_html=True)
st.title("Portfolio Risk Simulator")
st.markdown(
    '<div class="muted">multivariate-normal log returns · cholesky · '
    "VaR · expected shortfall · moments</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Simulation parameters")
    paths = st.slider("Simulated paths", 1_000, 200_000, 50_000, step=1_000)
    horizon = st.number_input("Horizon (trading days)", min_value=1, value=10, step=1)
    pv = st.number_input("Portfolio value ($)", min_value=0, value=1_000_000, step=10_000)
    seed_on = st.checkbox("Fix random seed (reproducible)", value=True)
    seed = 42 if seed_on else None
 
    with st.expander("Advanced model settings"):
        rf_pct = st.number_input(
            "Risk-free rate (% / yr)", value=4.0, step=0.25,
            help="Used for Sharpe, Sortino, and the tangency portfolio.",
        )
        drift = st.selectbox(
            "Return drift convention", ["log", "arithmetic"], index=0,
            help="'arithmetic' treats your input as the arithmetic expected "
            "return and applies the GBM correction m − ½σ².",
        )
        repair = st.selectbox(
            "Correlation matrix repair", ["nearest", "clip", "none"], index=0,
            help="How to fix a non-positive-definite matrix: Higham nearest "
            "(most faithful), eigenvalue clip (fast), or none.",
        )
        stepped_on = st.checkbox(
            "Simulate stepped paths (enables drawdown)", value=True,
            help="Steps day-by-day to build equity curves; capped at 20,000 paths.",
        )
 
    st.divider()
    st.subheader("Fetch by ticker")
    st.caption("Enter your own symbols; prices are pulled from Yahoo Finance.")
    tickers_str = st.text_input(
        "Tickers (comma-separated)", value="AAPL, MSFT, NVDA, TLT, GLD"
    )
    period = st.selectbox("History length", ["1y", "2y", "5y", "10y"], index=1)
    if st.button("Fetch & estimate →", width="stretch"):
        try:
            from risk import fetch_prices
 
            with st.spinner("Downloading prices…"):
                prices = fetch_prices(tickers_str.split(","), period=period)
            est = estimate_from_prices(prices)
            st.session_state.assets = pd.DataFrame(
                {
                    "Asset": est.names,
                    "Return %/yr": np.round(est.annual_ret_pct, 2),
                    "Vol %/yr": np.round(est.annual_vol_pct, 2),
                    "Weight %": np.round(np.full(len(est.names), 100 / len(est.names)), 2),
                }
            )
            st.session_state.corr = est.corr
            st.session_state.result = None
            st.success(
                f"Loaded {len(est.names)} tickers · {est.observations} daily returns "
                f"over the last {period}."
            )
        except Exception as exc: 
            st.error(str(exc))
 
    st.divider()
    st.subheader("From a price file")
    st.caption(
        "Or upload/paste a price table (one column per asset, oldest row first) "
        "to estimate returns, volatilities, and the correlation matrix."
    )
    upload = st.file_uploader("Price CSV", type=["csv", "tsv"], label_visibility="collapsed")
    pasted = st.text_area("…or paste CSV/TSV", height=120, placeholder="Date,SPY,AGG,GLD\n...")
 
    if st.button("Estimate covariance →", width="stretch"):
        try:
            if upload is not None:
                raw = pd.read_csv(upload, sep=None, engine="python")
            elif pasted.strip():
                from io import StringIO
 
                raw = pd.read_csv(StringIO(pasted), sep=None, engine="python")
            else:
                raise ValueError("Upload a file or paste a table first.")
            est = estimate_from_prices(raw)
            st.session_state.assets = pd.DataFrame(
                {
                    "Asset": est.names,
                    "Return %/yr": np.round(est.annual_ret_pct, 2),
                    "Vol %/yr": np.round(est.annual_vol_pct, 2),
                    "Weight %": np.round(np.full(len(est.names), 100 / len(est.names)), 2),
                }
            )
            st.session_state.corr = est.corr
            st.session_state.result = None
            st.success(f"Loaded {len(est.names)} assets from {est.observations} observations.")
        except Exception as exc: 
            st.error(str(exc))
 
left, right = st.columns([1.05, 0.95], gap="large")
 
with left:
    st.markdown("#### Assets & weights")
    assets = st.data_editor(
        st.session_state.assets,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="asset_editor",
        column_config={
            "Return %/yr": st.column_config.NumberColumn(format="%.2f"),
            "Vol %/yr": st.column_config.NumberColumn(format="%.2f"),
            "Weight %": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    assets = assets.dropna(subset=["Asset"]).reset_index(drop=True)
    st.session_state.assets = assets
    names = assets["Asset"].astype(str).tolist()
    n = len(names)
    sync_corr_size(n, names)
 
    wsum = float(pd.to_numeric(assets["Weight %"], errors="coerce").fillna(0).sum())
    note = "" if abs(wsum - 100) < 0.01 else " — normalized at run"
    st.markdown(f'<div class="muted">Σ weights = {wsum:.1f}%{note}</div>', unsafe_allow_html=True)
 
with right:
    st.markdown("#### Correlation matrix")
    st.caption("Edit any cell; the matrix is re-symmetrized and clamped on run.")
    corr_df = pd.DataFrame(st.session_state.corr, index=names, columns=names)
    edited = st.data_editor(
        corr_df,
        width="stretch",
        key="corr_editor",
    )
    st.session_state.corr = edited.to_numpy(dtype=float)

run = st.button("Run simulation", type="primary", width="stretch")
 
if run:
    try:
        a = st.session_state.assets

        c = st.session_state.corr.astype(float)
        c = (c + c.T) / 2.0
        c = np.clip(c, -0.999, 0.999)
        np.fill_diagonal(c, 1.0)
        st.session_state.corr = c
 
        ret_arr = pd.to_numeric(a["Return %/yr"], errors="coerce").fillna(0).to_numpy()
        vol_arr = pd.to_numeric(a["Vol %/yr"], errors="coerce").fillna(0).to_numpy()
        w_arr = pd.to_numeric(a["Weight %"], errors="coerce").fillna(0).to_numpy()
 
        res = run_simulation(
            returns_pct=ret_arr,
            vols_pct=vol_arr,
            weights=w_arr,
            corr=c,
            paths=paths,
            horizon_days=int(horizon),
            portfolio_val=float(pv),
            seed=seed,
            drift=drift,
            repair=repair,
        )
        st.session_state.result = res

        ppy = TRADING_DAYS / int(horizon)
        rf_period = (rf_pct / 100.0) / ppy
        st.session_state.perf = {
            "sharpe": sharpe_ratio(res.returns, rf=rf_period, periods_per_year=ppy),
            "sortino": sortino_ratio(res.returns, target=0.0, periods_per_year=ppy),
        }
 
        if stepped_on:
            st.session_state.steps = simulate_paths(
                returns_pct=ret_arr, vols_pct=vol_arr, weights=w_arr, corr=c,
                paths=paths, horizon_days=int(horizon), drift=drift,
                repair=repair, seed=seed,
            )
        else:
            st.session_state.steps = None
 
        st.session_state.warning = (
            "Correlation matrix wasn't positive-definite; a small adjustment was applied."
            if res.jitter > 0
            else None
        )
    except Exception as exc: 
        st.session_state.result = None
        st.session_state.warning = None
        st.error(str(exc))

res = st.session_state.result
if st.session_state.warning:
    st.warning(st.session_state.warning)
 
if res is None:
    st.info("Configure the portfolio and press **Run simulation** to see the risk report.")
else:
    tab_risk, tab_paths, tab_opt = st.tabs(
        ["Risk report", "Paths & drawdown", "Optimization"]
    )

    with tab_risk:
        r95, r99 = res.level(0.95), res.level(0.99)
        st.markdown(f"#### {res.horizon_days}-day horizon · {res.paths:,} paths")
        c1, c2, c3 = st.columns(3)
        c1.metric("VaR 95%", f"${r95.var_money:,.0f}", f"{r95.var_return*100:.2f}% of value", delta_color="off")
        c2.metric("VaR 99%", f"${r99.var_money:,.0f}", f"{r99.var_return*100:.2f}% of value", delta_color="off")
        c3.metric("Prob. of loss", f"{res.prob_loss*100:.1f}%", f"expected {res.moments['mean']*100:.2f}%", delta_color="off")
        c4, c5, c6 = st.columns(3)
        c4.metric("Expected Shortfall 95%", f"${r95.es_money:,.0f}", f"{r95.es_return*100:.2f}% of value", delta_color="off")
        c5.metric("Expected Shortfall 99%", f"${r99.es_money:,.0f}", f"{r99.es_return*100:.2f}% of value", delta_color="off")
        c6.metric("Volatility (horizon)", f"{res.vol_horizon*100:.2f}%", f"{res.vol_annual*100:.2f}% annualized", delta_color="off")
 
        perf = st.session_state.perf or {"sharpe": 0.0, "sortino": 0.0}
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Sharpe (ann.)", f"{perf['sharpe']:.2f}")
        p2.metric("Sortino (ann.)", f"{perf['sortino']:.2f}")
        p3.metric("Skewness", f"{res.moments['skew']:.3f}")
        p4.metric("Excess kurtosis", f"{res.moments['kurt']:.3f}")
 
        centers, counts = histogram(res.returns)
        var95_pct = -r95.var_return * 100
        es95_pct = -r95.es_return * 100
        centers_pct = centers * 100
        colors = [LOSS if c <= var95_pct else (GAIN if c >= 0 else "#6A4A3F") for c in centers_pct]
        fig = go.Figure(go.Bar(x=centers_pct, y=counts, marker_color=colors, marker_line_width=0))
        for x, color, label in [(var95_pct, LOSS, "VaR 95"), (es95_pct, AMBER, "ES 95")]:
            fig.add_vline(x=x, line_dash="dash", line_color=color, annotation_text=label, annotation_font_color=color)
        fig.add_vline(x=0, line_color=INK_FAINT)
        fig.update_layout(
            title="Portfolio return distribution — tail beyond 95% VaR shaded",
            xaxis_title="Return over horizon (%)", yaxis_title="Paths",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=INK_DIM, bargap=0, height=360, margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(gridcolor=GRID, zeroline=False)
        fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")

    with tab_paths:
        steps = st.session_state.steps
        if steps is None:
            st.info("Enable **Simulate stepped paths** in *Advanced model settings* and re-run to see equity curves and drawdown.")
        else:
            dd = steps.drawdowns
            med_dd, p95_dd = float(np.median(dd)), float(np.percentile(dd, 95))
            ann = TRADING_DAYS / steps.steps
            med_final = float(np.median(steps.final_returns))
            ann_ret = (1 + med_final) ** ann - 1
            calmar = calmar_ratio(ann_ret, med_dd)
            d1, d2, d3 = st.columns(3)
            d1.metric("Median max drawdown", f"{med_dd*100:.1f}%")
            d2.metric("95th-pct max drawdown", f"{p95_dd*100:.1f}%")
            d3.metric("Calmar (median path)", f"{calmar:.2f}")
 
            t = np.arange(steps.steps + 1)
            p5, p50, p95 = np.percentile(steps.equity, [5, 50, 95], axis=0)
            fan = go.Figure()
            fan.add_trace(go.Scatter(x=t, y=(p95 - 1) * 100, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fan.add_trace(go.Scatter(x=t, y=(p5 - 1) * 100, fill="tonexty", fillcolor="rgba(62,125,166,0.20)",
                                     line=dict(width=0), name="5–95%"))
            rng = np.random.default_rng(0)
            for i in rng.choice(steps.paths, size=min(50, steps.paths), replace=False):
                fan.add_trace(go.Scatter(x=t, y=(steps.equity[i] - 1) * 100, line=dict(width=0.5, color="rgba(233,227,213,0.18)"),
                                         showlegend=False, hoverinfo="skip"))
            fan.add_trace(go.Scatter(x=t, y=(p50 - 1) * 100, line=dict(width=2, color=AMBER), name="median"))
            fan.update_layout(
                title="Simulated equity curves (cumulative return)",
                xaxis_title="Trading day", yaxis_title="Cumulative return (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color=INK_DIM, height=320, margin=dict(l=10, r=10, t=50, b=10),
            )
            fan.update_xaxes(gridcolor=GRID)
            fan.update_yaxes(gridcolor=GRID)
            st.plotly_chart(fan, width="stretch")
 
            dc, de = histogram(dd * 100, bins=50)
            ddfig = go.Figure(go.Bar(x=dc, y=de, marker_color=LOSS, marker_line_width=0, opacity=0.85))
            ddfig.add_vline(x=med_dd * 100, line_dash="dash", line_color=AMBER, annotation_text="median")
            ddfig.update_layout(
                title="Maximum drawdown distribution across paths",
                xaxis_title="Max drawdown (%)", yaxis_title="Paths",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color=INK_DIM, bargap=0, height=280, margin=dict(l=10, r=10, t=50, b=10),
            )
            ddfig.update_xaxes(gridcolor=GRID)
            ddfig.update_yaxes(gridcolor=GRID)
            st.plotly_chart(ddfig, width="stretch")

    with tab_opt:
        st.caption("Mean-variance optimization from the current expected returns, "
                   "volatilities, and correlation matrix. Long-only adds w ≥ 0, Σw = 1.")
        a = st.session_state.assets
        mu = pd.to_numeric(a["Return %/yr"], errors="coerce").fillna(0).to_numpy() / 100.0
        vols = pd.to_numeric(a["Vol %/yr"], errors="coerce").fillna(0).to_numpy() / 100.0
        cur_w = pd.to_numeric(a["Weight %"], errors="coerce").fillna(0).to_numpy()
        cur_w = cur_w / (cur_w.sum() or 1.0)
        cmat = st.session_state.corr.astype(float)
        cov = cmat * np.outer(vols, vols)
        rf = rf_pct / 100.0
 
        long_only = st.checkbox("Long-only (no short positions)", value=True)
        mv = min_variance_weights(cov, long_only=long_only)
        ms = max_sharpe_weights(mu, cov, rf=rf, long_only=long_only)
        cr, cv, cs = portfolio_stats(cur_w, mu, cov, rf)
 
        comp = pd.DataFrame(
            {
                "Asset": a["Asset"].tolist(),
                "Current %": np.round(cur_w * 100, 1),
                "Min-variance %": np.round(mv.weights * 100, 1),
                "Max-Sharpe %": np.round(ms.weights * 100, 1),
            }
        )
        st.dataframe(comp, width="stretch", hide_index=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Current", f"Sharpe {cs:.2f}", f"{cr*100:.1f}% ret · {cv*100:.1f}% vol", delta_color="off")
        mv_sharpe = portfolio_stats(mv.weights, mu, cov, rf)[2]
        s2.metric("Min-variance", f"Sharpe {mv_sharpe:.2f}",
                  f"{mv.ret*100:.1f}% ret · {mv.vol*100:.1f}% vol", delta_color="off")
        s3.metric("Max-Sharpe", f"Sharpe {ms.sharpe:.2f}", f"{ms.ret*100:.1f}% ret · {ms.vol*100:.1f}% vol", delta_color="off")
 
        b1, b2 = st.columns(2)
        if b1.button("Apply min-variance weights", width="stretch"):
            st.session_state.assets["Weight %"] = np.round(mv.weights * 100, 2)
            st.rerun()
        if b2.button("Apply max-Sharpe weights", width="stretch"):
            st.session_state.assets["Weight %"] = np.round(ms.weights * 100, 2)
            st.rerun()
 
        # efficient frontier with the three portfolios marked
        fr_ret, fr_vol = efficient_frontier(mu, cov, points=60)
        ef = go.Figure()
        ef.add_trace(go.Scatter(x=fr_vol * 100, y=fr_ret * 100, line=dict(color=INK_DIM, width=2), name="frontier"))
        ef.add_trace(go.Scatter(x=[cv * 100], y=[cr * 100], mode="markers", marker=dict(color=INK, size=11, symbol="circle"), name="current"))
        ef.add_trace(go.Scatter(x=[mv.vol * 100], y=[mv.ret * 100], mode="markers", marker=dict(color=COOL, size=12, symbol="diamond"), name="min-variance"))
        ef.add_trace(go.Scatter(x=[ms.vol * 100], y=[ms.ret * 100], mode="markers", marker=dict(color=AMBER, size=13, symbol="star"), name="max-Sharpe"))
        ef.update_layout(
            title="Efficient frontier (annualized)",
            xaxis_title="Volatility (%)", yaxis_title="Expected return (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=INK_DIM, height=360, margin=dict(l=10, r=10, t=50, b=10),
        )
        ef.update_xaxes(gridcolor=GRID)
        ef.update_yaxes(gridcolor=GRID)
        st.plotly_chart(ef, width="stretch")
        if long_only and not ms.long_only_applied:
            st.caption("⚠ SciPy not available — showing the unconstrained (short-allowed) solution.")

st.markdown("#### Correlation heatmap")
heat = go.Figure(
    go.Heatmap(
        z=st.session_state.corr,
        x=names,
        y=names,
        colorscale=[[0.0, COOL], [0.5, PANEL], [1.0, AMBER]],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(st.session_state.corr, 2),
        texttemplate="%{text}",
        textfont={"family": "ui-monospace, monospace", "size": 11},
        showscale=True,
    )
)
heat.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color=INK_DIM,
    height=90 + 46 * len(names),
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_autorange="reversed",
)
st.plotly_chart(heat, width="stretch")
 
st.markdown(
    '<div class="muted">Method: Σ = ρ ∘ (σσᵀ) is Cholesky-factored as Σ = LLᵀ. '
    "Each path draws z ~ N(0, I); horizon log returns are μh + √h·Lz; asset "
    "simple returns eʳ − 1 are weighted into a portfolio return. VaR is the loss "
    "at the chosen confidence; expected shortfall is the mean loss in that tail. "
    "Educational tool — not investment advice.</div>",
    unsafe_allow_html=True,
)
    