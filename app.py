import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize
from scipy.stats import norm, beta as beta_dist_fn
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# ฟังก์ชันคณิตศาสตร์ (Core Logic - โค้ดเดิมจากของคุณ)
# ═══════════════════════════════════════════════════════════════════════════════
def from_u(u, dist, p1, p2, p3=None, p4=None):
    if dist == "Normal":
        return p1 + p2 * u
    elif dist == "Lognormal":
        mu_ln = math.log(p1**2 / math.sqrt(p1**2 + p2**2))
        sig_ln = math.sqrt(math.log(1 + (p2 / p1) ** 2))
        return math.exp(mu_ln + sig_ln * u)
    elif dist == "Tanh":
        p = max(1e-9, min(1 - 1e-9, norm.cdf(u)))
        v = 0.5 + math.atanh(2 * p - 1) / p2
        return p3 + v * (p4 - p3)
    elif dist == "BetaDist":
        p = max(1e-9, min(1 - 1e-9, norm.cdf(u)))
        u_01 = beta_dist_fn.ppf(p, p1, p2)
        return p3 + u_01 * (p4 - p3)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

def dist_mean(dist, p1, p2, p3=None, p4=None):
    return from_u(0, dist, p1, p2, p3, p4)

def performance_function(rv):
    pm, u, sh, sH, c, tanf = rv
    return c + (sh - pm - u) * tanf - (sH - sh)

def run_form(variables, corr_matrix):
    n = len(variables)
    dists = [v["dist"] for v in variables]
    params = [(v["p1"], v["p2"], v.get("p3"), v.get("p4")) for v in variables]

    R = np.array(corr_matrix, dtype=float)
    R = (R + R.T) / 2
    R += np.eye(n) * 1e-8
    L = np.linalg.cholesky(R)

    x_mean = [from_u(0, dists[i], *params[i]) for i in range(n)]
    g_rvo = performance_function(x_mean)

    def g_of_z(z):
        uc = L @ z
        x = [from_u(float(uc[i]), dists[i], *params[i]) for i in range(n)]
        return performance_function(x)

    result = minimize(
        lambda z: float(np.dot(z, z)),
        np.zeros(n),
        method="SLSQP",
        constraints=[{"type": "eq", "fun": g_of_z}],
        bounds=[(-10, 10)] * n,
        options={"ftol": 1e-14, "maxiter": 5000, "disp": False},
    )

    z_star = result.x
    beta = math.sqrt(max(0.0, float(np.dot(z_star, z_star))))
    if g_rvo < 0:
        beta = -beta

    Pf = norm.cdf(-beta)
    u_star = L @ z_star
    x_star = [from_u(float(u_star[i]), dists[i], *params[i]) for i in range(n)]
    g_dp = performance_function(x_star)
    alpha = z_star / (beta if abs(beta) > 1e-9 else 1.0)

    return {
        "beta": beta, "Pf": Pf, "z_star": z_star.tolist(),
        "u_star": u_star.tolist(), "x_star": x_star, "alpha": alpha.tolist(),
        "g_rvo": g_rvo, "g_dp": g_dp, "converged": result.success,
    }

def run_monte_carlo(variables, corr_matrix, n_samples, seed=42):
    n = len(variables)
    dists = [v["dist"] for v in variables]
    params = [(v["p1"], v["p2"], v.get("p3"), v.get("p4")) for v in variables]

    rng = np.random.default_rng(seed)
    R = np.array(corr_matrix, dtype=float)
    R = (R + R.T) / 2
    R += np.eye(n) * 1e-8
    L = np.linalg.cholesky(R)

    z_mat = rng.standard_normal((n, n_samples))
    u_mat = L @ z_mat

    samples = np.zeros((n, n_samples))
    for i in range(n):
        for j in range(n_samples):
            try:
                samples[i, j] = from_u(float(u_mat[i, j]), dists[i], *params[i])
            except:
                samples[i, j] = dist_mean(dists[i], *params[i])

    g = (samples[4] + (samples[2] - samples[0] - samples[1]) * samples[5] - (samples[3] - samples[2]))
    failures = int(np.sum(g < 0))
    Pf = failures / n_samples
    beta_eq = -norm.ppf(Pf) if 0 < Pf < 1 else (10 if Pf == 0 else -10)

    return {"Pf": Pf, "beta_eq": beta_eq, "failures": failures, "n": n_samples, "g_vals": g}

# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Wellbore Stability", layout="wide", page_icon="⬡")

st.title("⬡ Vertical Wellbore Stability")
st.markdown("**FORM + Monte Carlo Reliability Analysis**")

# Default Data
default_vars = [
    {"name": "Mud pressure", "sym": "pm", "dist": "Normal", "p1": 10.0, "p2": 4.0, "p3": None, "p4": None},
    {"name": "Pore pressure", "sym": "u", "dist": "Lognormal", "p1": 20.0, "p2": 9.0, "p3": None, "p4": None},
    {"name": "Min horizontal stress", "sym": "σh", "dist": "Tanh", "p1": 0.0, "p2": 3.0, "p3": 10.0, "p4": 20.0},
    {"name": "Max horizontal stress", "sym": "σH", "dist": "BetaDist", "p1": 3.0, "p2": 3.0, "p3": 20.0, "p4": 40.0},
    {"name": "Cohesion", "sym": "c'", "dist": "Lognormal", "p1": 36.0, "p2": 7.2, "p3": None, "p4": None},
    {"name": "Friction angle tan(φ')", "sym": "tan(φ')", "dist": "Lognormal", "p1": 0.577, "p2": 0.05774, "p3": None, "p4": None},
]
df_vars = pd.DataFrame(default_vars)

default_corr = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, -0.5],
    [0.0, 0.0, 0.0, 0.0, -0.5, 1.0],
]
syms = [v["sym"] for v in default_vars]
df_corr = pd.DataFrame(default_corr, columns=syms, index=syms)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Variables", "Correlation", "FORM Results", "Monte Carlo", "Charts"])

with tab1:
    st.subheader("Define random variables, distribution types, and parameters.")
    st.info("Performance function: $g = c' + (σ_h - p_m - u) \\cdot \\tan(\\phi') - (σ_H - σ_h)$\n\n$g > 0 \\rightarrow$ Stable | $g < 0 \\rightarrow$ Failure")
    edited_vars_df = st.data_editor(df_vars, num_rows="dynamic", use_container_width=True)

with tab2:
    st.subheader("Correlation Matrix")
    st.caption("Enter the correlation matrix. Default correlation between c' and tan(φ') = −0.5.")
    edited_corr_df = st.data_editor(df_corr, use_container_width=True)

variables_list = edited_vars_df.to_dict('records')
corr_matrix = edited_corr_df.to_numpy()

# Initialize Session State
if 'form_res' not in st.session_state: st.session_state.form_res = None
if 'mc_res' not in st.session_state: st.session_state.mc_res = None

with tab3:
    if st.button("▶ Run FORM Analysis", type="primary"):
        with st.spinner('Running FORM Optimization...'):
            try:
                st.session_state.form_res = run_form(variables_list, corr_matrix)
                st.success("FORM Analysis Converged!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.form_res:
        res = st.session_state.form_res
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reliability Index β", f"{res['beta']:.4f}")
        col2.metric("Probability of Failure Pf", f"{res['Pf']*100:.3f}%")
        col3.metric("g (Mean values, RVO)", f"{res['g_rvo']:.3f}")
        col4.metric("g (Design Point)", f"{res['g_dp']:.2e}")

        st.subheader("Design Point & Importance Factors")
        dp_data = []
        for i, v in enumerate(variables_list):
            rvo = from_u(0, v["dist"], v["p1"], v["p2"], v.get("p3"), v.get("p4"))
            dp_data.append({
                "Variable": v["name"], "Symbol": v["sym"], "Dist": v["dist"],
                "Real Space x*": res['x_star'][i], "Norm Space u*": res['u_star'][i],
                "Mean (RVO)": rvo, "α (Importance)": res['alpha'][i]
            })
        st.dataframe(pd.DataFrame(dp_data), use_container_width=True)

with tab4:
    col1, col2 = st.columns([1, 3])
    with col1:
        n_samples = st.number_input("Samples", min_value=1000, value=100000, step=10000)
        seed = st.number_input("Seed", value=42)
        if st.button("◎ Run Monte Carlo"):
            with st.spinner('Running Simulation...'):
                try:
                    st.session_state.mc_res = run_monte_carlo(variables_list, corr_matrix, int(n_samples), int(seed))
                    st.success("Simulation Complete!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.session_state.mc_res:
            mc = st.session_state.mc_res
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Pf (Monte Carlo)", f"{mc['Pf']*100:.3f}%")
            mc2.metric("Equivalent β", f"{mc['beta_eq']:.4f}")
            mc3.metric("Failures / Samples", f"{mc['failures']:,} / {mc['n']:,}")
            
            if st.session_state.form_res:
                st.caption(f"FORM β = {st.session_state.form_res['beta']:.4f} | MC β = {mc['beta_eq']:.4f} | Δ = {abs(st.session_state.form_res['beta'] - mc['beta_eq']):.4f}")

with tab5:
    st.subheader("Analysis Charts")
    if not st.session_state.form_res and not st.session_state.mc_res:
        st.info("Run FORM or Monte Carlo first to generate charts.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.form_res:
            res = st.session_state.form_res
            alpha = res["alpha"]
            imp = [a**2 for a in alpha]
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#58a6ff" if a > 0 else "#f85149" for a in alpha]
            ax.barh(syms, imp, color=colors)
            ax.set_xlabel("Importance factor α²")
            ax.set_title(f"Sensitivity / Importance Factors (β = {res['beta']:.4f})")
            st.pyplot(fig)
            
    with col2:
        if st.session_state.mc_res:
            g = st.session_state.mc_res["g_vals"]
            Pf = st.session_state.mc_res["Pf"]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.hist(g[g >= 0], bins=80, color="#58a6ff", alpha=0.7, label="Stable (g ≥ 0)")
            ax2.hist(g[g < 0], bins=40, color="#f85149", alpha=0.8, label=f"Failure (g < 0)")
            ax2.axvline(0, color="black", linestyle="--")
            ax2.set_xlabel("Performance function g")
            ax2.set_title(f"Distribution of g (Pf = {Pf*100:.3f}%)")
            ax2.legend()
            st.pyplot(fig2)