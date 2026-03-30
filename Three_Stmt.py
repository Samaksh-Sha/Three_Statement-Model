# ═══════════════════════════════════════════════════════════════════════════
#  Three-Statement Integrated Model  |  Samaksh Sha  |  FLAME University
#  IS → BS → CFS fully linked  ·  5 pre-fitted companies  ·  yfinance custom
#  Balance check enforced  ·  Full ratio analysis  ·  Explanations on every line
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="3-Statement Model · Samaksh Sha",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]    { background:#0c0c0c; border-right:1px solid #1e1e1e; }
[data-testid="stSidebar"] *  { color:#e2e8f0 !important; }
.metric-card { background:#111; border:1px solid #1e1e1e; border-radius:8px;
               padding:14px 16px; text-align:center; }
.m-label { font-size:10px; color:#64748b; text-transform:uppercase;
            letter-spacing:.08em; margin-bottom:4px; }
.m-value { font-size:20px; font-weight:700; }
.m-sub   { font-size:11px; color:#475569; margin-top:2px; }
.sec     { font-size:10px; font-weight:600; text-transform:uppercase;
            letter-spacing:.09em; color:#475569; border-bottom:1px solid #1e1e1e;
            padding-bottom:5px; margin:18px 0 10px; }
.explain-box { background:#0d1117; border:1px solid #21262d; border-radius:8px;
               padding:14px 16px; margin:8px 0 14px; font-size:12px;
               color:#8b949e; line-height:1.7; }
.explain-box b  { color:#58a6ff; font-weight:600; }
.explain-box .calc { background:#161b22; border-radius:4px; padding:3px 8px;
                     font-family:monospace; color:#79c0ff; font-size:11px;
                     display:inline-block; margin:2px 0; }
.explain-box ul { margin:5px 0 5px 16px; padding:0; }
.explain-box li { margin:3px 0; }
.bal-ok  { background:#0f2318; border:1px solid #238636; border-radius:6px;
           padding:8px 14px; font-size:12px; color:#86efac; }
.bal-err { background:#1a0909; border:1px solid #da3633; border-radius:6px;
           padding:8px 14px; font-size:12px; color:#fca5a5; }
.chart-exp { background:#0d1117; border-left:3px solid #388bfd; border-radius:0 6px 6px 0;
             padding:9px 13px; font-size:12px; color:#8b949e; margin:4px 0 12px; line-height:1.6; }
.stmt-header { font-size:13px; font-weight:700; color:#f1f5f9;
               border-bottom:2px solid #1e40af; padding-bottom:6px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ── THREE-STATEMENT ENGINE ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def build_model(p: dict, years: int = 5) -> dict:
    """
    Fully linked three-statement model.
    All values in ₹ Crores unless noted.

    Base-year inputs (prefix b_):
        b_revenue, b_gross_margin (%), b_ebitda_margin (%),
        b_da_pct (% rev), b_interest_expense, b_tax_rate (%),
        b_cash, b_receivables_days, b_inventory_days,
        b_payables_days, b_ppe_net, b_other_assets,
        b_short_debt, b_long_debt, b_other_liabilities, b_equity

    Projection assumptions:
        rev_growth (% CAGR), gross_margin_exit (%),
        ebitda_margin_exit (%), da_pct_exit (%),
        capex_pct (% rev), debt_repayment_pa (₹ Cr/yr),
        new_debt_raised (₹ Cr/yr), interest_rate (%),
        tax_rate (%), dividend_payout (% NI)
    """
    # ── Unpack ────────────────────────────────────────────────────────────
    tax   = p["tax_rate"] / 100
    div   = p["dividend_payout"] / 100

    # Year-0 BS — equity is the RESIDUAL plug so base-year BS balances exactly
    cash_0   = p["b_cash"]
    recv_0   = p["b_receivables_days"] * p["b_revenue"] / 365
    inv_0    = p["b_inventory_days"]   * p["b_revenue"] / 365
    pay_0    = p["b_payables_days"]    * p["b_revenue"] / 365
    ppe_0    = p["b_ppe_net"]
    oth_a_0  = p["b_other_assets"]
    s_debt_0 = p["b_short_debt"]
    l_debt_0 = p["b_long_debt"]
    oth_l_0  = p["b_other_liabilities"]
    total_assets_0 = cash_0 + recv_0 + inv_0 + ppe_0 + oth_a_0
    total_liab_0   = pay_0  + s_debt_0 + l_debt_0 + oth_l_0
    equity_0       = total_assets_0 - total_liab_0   # plug — guarantees BS balance

    # ── Projection containers ─────────────────────────────────────────────
    IS, BS, CF = {}, {}, {}
    for k in ["revenue","cogs","gross_profit","gross_margin",
              "opex","ebitda","da","ebit","interest","ebt","tax","ni",
              "ebitda_margin","ebit_margin","ni_margin"]:
        IS[k] = []
    for k in ["cash","receivables","inventory","ppe","other_assets","total_assets",
              "payables","short_debt","long_debt","other_liabilities","total_liabilities",
              "equity","retained_earnings","total_le","check"]:
        BS[k] = []
    for k in ["ni","da","delta_wc","cfo",
              "capex","cfi",
              "debt_raised","debt_repaid","dividends","cff",
              "net_change","beg_cash","end_cash"]:
        CF[k] = []

    # State variables
    ret_earn  = 0.0        # accumulated retained earnings (starts at 0)
    cash      = cash_0
    recv      = recv_0
    inv       = inv_0
    pay       = pay_0
    ppe       = ppe_0
    oth_a     = oth_a_0
    s_debt    = s_debt_0
    l_debt    = l_debt_0
    oth_l     = oth_l_0
    equity    = equity_0
    prev_rev  = p["b_revenue"]
    prev_recv = recv_0
    prev_inv  = inv_0
    prev_pay  = pay_0

    for yr in range(1, years + 1):
        frac = yr / years   # linear interpolation factor

        # ── Income Statement ──────────────────────────────────────────
        rev = p["b_revenue"] * (1 + p["rev_growth"] / 100) ** yr

        gm    = p["b_gross_margin"] + (p["gross_margin_exit"] - p["b_gross_margin"]) * frac
        em    = p["b_ebitda_margin"]+ (p["ebitda_margin_exit"] - p["b_ebitda_margin"])* frac
        da_p  = p["b_da_pct"]       + (p["da_pct_exit"]        - p["b_da_pct"])       * frac

        gross  = rev * gm   / 100
        cogs   = rev - gross
        ebitda = rev * em   / 100
        opex   = gross - ebitda            # SGA = Gross − EBITDA
        da     = rev * da_p / 100
        ebit   = ebitda - da

        # Interest on average debt (simple)
        avg_debt    = s_debt + l_debt
        int_exp     = avg_debt * p["interest_rate"] / 100
        ebt         = ebit - int_exp
        tax_charge  = max(0.0, ebt * tax)
        ni          = ebt - tax_charge

        IS["revenue"].append(rev);      IS["cogs"].append(cogs)
        IS["gross_profit"].append(gross); IS["gross_margin"].append(gm)
        IS["opex"].append(opex);        IS["ebitda"].append(ebitda)
        IS["da"].append(da);            IS["ebit"].append(ebit)
        IS["interest"].append(int_exp); IS["ebt"].append(ebt)
        IS["tax"].append(tax_charge);   IS["ni"].append(ni)
        IS["ebitda_margin"].append(em); IS["ebit_margin"].append(ebit/rev*100)
        IS["ni_margin"].append(ni/rev*100)

        # ── Cash Flow Statement ────────────────────────────────────────
        beg_cash = cash

        # Working capital (using days × revenue / 365)
        new_recv = p["b_receivables_days"] * rev / 365
        new_inv  = p["b_inventory_days"]   * rev / 365
        new_pay  = p["b_payables_days"]    * rev / 365

        d_recv = new_recv - prev_recv
        d_inv  = new_inv  - prev_inv
        d_pay  = new_pay  - prev_pay
        delta_wc = -(d_recv + d_inv - d_pay)  # ↑ WC = cash outflow

        cfo = ni + da + delta_wc

        capex = rev * p["capex_pct"] / 100
        cfi   = -capex

        new_debt  = p.get("new_debt_raised", 0.0)
        rep_debt  = min(p["debt_repayment_pa"], max(0.0, l_debt))  # can't repay more than outstanding
        dividends = max(0.0, ni * div)
        cff       = new_debt - rep_debt - dividends

        net_change = cfo + cfi + cff
        end_cash   = beg_cash + net_change

        CF["ni"].append(ni);        CF["da"].append(da)
        CF["delta_wc"].append(delta_wc); CF["cfo"].append(cfo)
        CF["capex"].append(-capex); CF["cfi"].append(cfi)
        CF["debt_raised"].append(new_debt); CF["debt_repaid"].append(-rep_debt)
        CF["dividends"].append(-dividends); CF["cff"].append(cff)
        CF["net_change"].append(net_change)
        CF["beg_cash"].append(beg_cash); CF["end_cash"].append(end_cash)

        # ── Balance Sheet ──────────────────────────────────────────────
        # Update state
        cash   = max(0.0, end_cash)
        recv   = new_recv
        inv    = new_inv
        pay    = new_pay
        ppe    = max(0.0, ppe - da + capex)      # ppe + capex - depreciation
        # oth_a stays constant — any growth would require explicit financing source
        l_debt = max(0.0, l_debt + new_debt - rep_debt)
        # oth_l stays constant — changes require explicit modelling

        # Equity: opening + NI − dividends
        retained = ni - dividends
        ret_earn += retained
        equity   = equity_0 + ret_earn          # simplified: initial equity + cumul retained

        total_assets = cash + recv + inv + ppe + oth_a
        total_liab   = pay + s_debt + l_debt + oth_l
        total_le     = total_liab + equity
        check        = total_assets - total_le  # should be ≈ 0

        BS["cash"].append(cash);           BS["receivables"].append(recv)
        BS["inventory"].append(inv);       BS["ppe"].append(ppe)
        BS["other_assets"].append(oth_a);  BS["total_assets"].append(total_assets)
        BS["payables"].append(pay);        BS["short_debt"].append(s_debt)
        BS["long_debt"].append(l_debt);    BS["other_liabilities"].append(oth_l)
        BS["total_liabilities"].append(total_liab)
        BS["equity"].append(equity);       BS["retained_earnings"].append(ret_earn)
        BS["total_le"].append(total_le);   BS["check"].append(check)

        prev_rev  = rev
        prev_recv = new_recv
        prev_inv  = new_inv
        prev_pay  = new_pay

    return {"IS": IS, "BS": BS, "CF": CF, "years": years}


def compute_ratios(m: dict, p: dict) -> pd.DataFrame:
    """Derive key financial ratios from the model."""
    IS, BS = m["IS"], m["BS"]
    yrs = m["years"]
    labels = [f"Y{i+1}" for i in range(yrs)]

    data = {
        "Gross margin %":        [f"{v:.1f}%" for v in IS["gross_margin"]],
        "EBITDA margin %":       [f"{v:.1f}%" for v in IS["ebitda_margin"]],
        "Net margin %":          [f"{v:.1f}%" for v in IS["ni_margin"]],
        "EBIT margin %":         [f"{v:.1f}%" for v in IS["ebit_margin"]],
        "Interest coverage (x)": [f"{IS['ebitda'][i]/max(IS['interest'][i],0.01):.1f}x" for i in range(yrs)],
        "ROE %":                 [f"{IS['ni'][i]/BS['equity'][i]*100:.1f}%" for i in range(yrs)],
        "ROA %":                 [f"{IS['ni'][i]/BS['total_assets'][i]*100:.1f}%" for i in range(yrs)],
        "Debt/EBITDA (x)":       [f"{(BS['short_debt'][i]+BS['long_debt'][i])/max(IS['ebitda'][i],0.01):.1f}x" for i in range(yrs)],
        "Current ratio (x)":     [f"{(BS['cash'][i]+BS['receivables'][i]+BS['inventory'][i])/max(BS['payables'][i]+BS['short_debt'][i],0.01):.1f}x" for i in range(yrs)],
        "FCF (₹ Cr)":            [f"₹{IS['ni'][i]+IS['da'][i]-abs(m['CF']['capex'][i]):.0f}" for i in range(yrs)],
        "Capex/Revenue %":       [f"{abs(m['CF']['capex'][i])/IS['revenue'][i]*100:.1f}%" for i in range(yrs)],
        "Cash conversion %":     [f"{m['CF']['cfo'][i]/max(IS['ebitda'][i],0.01)*100:.0f}%" for i in range(yrs)],
    }
    return pd.DataFrame(data, index=labels).T


# ═══════════════════════════════════════════════════════════════════════════
# ── COMPANIES DATABASE ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

COMPANIES = {
    "Infosys": {
        "ticker": "INFY.NS", "flag": "💻", "sector": "IT Services",
        "desc": "India's second-largest IT services exporter. Asset-light model with exceptional FCF conversion.",
        "thesis": "Secular digital transformation spend, margin expansion from automation, return of capital via buybacks.",
        "tags": ["IT", "Asset-light", "High FCF"],
        # Base-year FY24 approximates
        "b_revenue": 153670.0, "b_gross_margin": 31.5, "b_ebitda_margin": 24.0,
        "b_da_pct": 2.5, "b_interest_expense": 320.0, "tax_rate": 25.17,
        "b_cash": 28000.0, "b_receivables_days": 68.0, "b_inventory_days": 2.0,
        "b_payables_days": 35.0, "b_ppe_net": 12000.0, "b_other_assets": 55000.0,
        "b_short_debt": 0.0, "b_long_debt": 2800.0, "b_other_liabilities": 22000.0,
        "b_equity": 72000.0,
        # Projections
        "rev_growth": 8.0, "gross_margin_exit": 33.0, "ebitda_margin_exit": 25.5,
        "da_pct_exit": 2.5, "capex_pct": 2.5, "debt_repayment_pa": 500.0,
        "new_debt_raised": 0.0, "interest_rate": 8.0, "dividend_payout": 60.0,
    },
    "Reliance Industries": {
        "ticker": "RELIANCE.NS", "flag": "⚡", "sector": "Diversified Conglomerate",
        "desc": "India's largest company by revenue. O2C, Jio telecom, Retail, and New Energy.",
        "thesis": "Jio 5G monetisation, retail premiumisation, green hydrogen optionality, refinery upgrade cycle.",
        "tags": ["Conglomerate", "Capital-intensive", "Growth"],
        "b_revenue": 921000.0, "b_gross_margin": 12.0, "b_ebitda_margin": 17.0,
        "b_da_pct": 3.5, "b_interest_expense": 18000.0, "tax_rate": 25.17,
        "b_cash": 185000.0, "b_receivables_days": 22.0, "b_inventory_days": 40.0,
        "b_payables_days": 55.0, "b_ppe_net": 480000.0, "b_other_assets": 220000.0,
        "b_short_debt": 45000.0, "b_long_debt": 260000.0, "b_other_liabilities": 95000.0,
        "b_equity": 580000.0,
        "rev_growth": 10.0, "gross_margin_exit": 13.5, "ebitda_margin_exit": 18.5,
        "da_pct_exit": 3.5, "capex_pct": 10.0, "debt_repayment_pa": 15000.0,
        "new_debt_raised": 20000.0, "interest_rate": 7.5, "dividend_payout": 10.0,
    },
    "Asian Paints": {
        "ticker": "ASIANPAINT.NS", "flag": "🎨", "sector": "Consumer / Paints",
        "desc": "India's largest paints company. 60%+ market share in decorative segment.",
        "thesis": "Housing upcycle, premiumisation of paint products, distribution network moat.",
        "tags": ["Consumer", "Moat", "Premium"],
        "b_revenue": 35000.0, "b_gross_margin": 42.0, "b_ebitda_margin": 19.0,
        "b_da_pct": 2.8, "b_interest_expense": 120.0, "tax_rate": 25.17,
        "b_cash": 4200.0, "b_receivables_days": 28.0, "b_inventory_days": 55.0,
        "b_payables_days": 42.0, "b_ppe_net": 8500.0, "b_other_assets": 9800.0,
        "b_short_debt": 200.0, "b_long_debt": 800.0, "b_other_liabilities": 5200.0,
        "b_equity": 16000.0,
        "rev_growth": 12.0, "gross_margin_exit": 44.0, "ebitda_margin_exit": 21.0,
        "da_pct_exit": 2.8, "capex_pct": 4.5, "debt_repayment_pa": 200.0,
        "new_debt_raised": 0.0, "interest_rate": 7.0, "dividend_payout": 55.0,
    },
    "Maruti Suzuki": {
        "ticker": "MARUTI.NS", "flag": "🚗", "sector": "Automobile",
        "desc": "India's largest passenger vehicle maker with ~41% market share.",
        "thesis": "SUV model cycle refresh, hybrid technology leadership, export ramp to developing markets.",
        "tags": ["Auto", "Cyclical", "Export"],
        "b_revenue": 138000.0, "b_gross_margin": 27.0, "b_ebitda_margin": 11.5,
        "b_da_pct": 3.5, "b_interest_expense": 250.0, "tax_rate": 25.17,
        "b_cash": 38000.0, "b_receivables_days": 8.0, "b_inventory_days": 20.0,
        "b_payables_days": 28.0, "b_ppe_net": 24000.0, "b_other_assets": 48000.0,
        "b_short_debt": 0.0, "b_long_debt": 0.0, "b_other_liabilities": 18000.0,
        "b_equity": 55000.0,
        "rev_growth": 9.0, "gross_margin_exit": 28.5, "ebitda_margin_exit": 13.0,
        "da_pct_exit": 3.5, "capex_pct": 5.5, "debt_repayment_pa": 0.0,
        "new_debt_raised": 0.0, "interest_rate": 7.0, "dividend_payout": 35.0,
    },
    "HDFC Bank": {
        "ticker": "HDFCBANK.NS", "flag": "🏦", "sector": "Banking",
        "desc": "India's largest private sector bank post HDFC merger.",
        "thesis": "HDFC mortgage cross-sell, CASA ratio improvement, NIM expansion.",
        "tags": ["Banking", "NBFC-merger", "Credit growth"],
        "b_revenue": 200000.0, "b_gross_margin": 70.0, "b_ebitda_margin": 35.0,
        "b_da_pct": 1.5, "b_interest_expense": 95000.0, "tax_rate": 25.17,
        "b_cash": 220000.0, "b_receivables_days": 15.0, "b_inventory_days": 0.0,
        "b_payables_days": 20.0, "b_ppe_net": 18000.0, "b_other_assets": 850000.0,
        "b_short_debt": 120000.0, "b_long_debt": 680000.0, "b_other_liabilities": 85000.0,
        "b_equity": 250000.0,
        "rev_growth": 14.0, "gross_margin_exit": 72.0, "ebitda_margin_exit": 37.0,
        "da_pct_exit": 1.5, "capex_pct": 0.8, "debt_repayment_pa": 10000.0,
        "new_debt_raised": 80000.0, "interest_rate": 7.2, "dividend_payout": 25.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# ── YFINANCE FETCH ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_data(ticker: str) -> dict:
    """Pull base-year financials from Yahoo Finance for custom model."""
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        fin  = t.financials
        bs   = t.balance_sheet
        cf   = t.cashflow

        def _c(df, key, div=1e7):
            if df is not None and not df.empty and key in df.index:
                v = df.loc[key].iloc[0]
                return round(float(v) / div, 0) if not pd.isna(v) else None
            return None

        revenue      = _c(fin, "Total Revenue")
        cogs         = _c(fin, "Cost Of Revenue")
        gross        = _c(fin, "Gross Profit")
        ebitda       = _c(fin, "EBITDA")
        ni           = _c(fin, "Net Income")
        int_exp      = _c(fin, "Interest Expense")
        da           = _c(cf,  "Depreciation And Amortization")
        capex        = _c(cf,  "Capital Expenditure")
        cash         = _c(bs,  "Cash And Cash Equivalents")
        recv         = _c(bs,  "Receivables")
        inv          = _c(bs,  "Inventory")
        ppe          = _c(bs,  "Net PPE")
        s_debt       = _c(bs,  "Current Debt")
        l_debt       = _c(bs,  "Long Term Debt")
        pay          = _c(bs,  "Payables")
        equity       = _c(bs,  "Stockholders Equity")

        gm   = gross  / revenue * 100 if (gross  and revenue) else 30.0
        em   = ebitda / revenue * 100 if (ebitda and revenue) else 15.0
        da_p = da     / revenue * 100 if (da     and revenue) else 3.0
        cap_p= abs(capex or 0) / (revenue or 1) * 100

        return dict(
            revenue=revenue or 5000.0, gross_margin=round(gm,1),
            ebitda_margin=round(em,1), da_pct=round(da_p,2),
            interest_expense=abs(int_exp or 0),
            cash=cash or 500.0, receivables=recv or 500.0,
            inventory=inv or 200.0, ppe=ppe or 2000.0,
            short_debt=s_debt or 0.0, long_debt=l_debt or 0.0,
            payables=pay or 400.0, equity=equity or 3000.0,
            capex_pct=round(cap_p, 1),
            # Derived days
            recv_days=round((recv or 500) / (revenue or 5000) * 365, 0),
            inv_days= round((inv  or 200) / (revenue or 5000) * 365, 0),
            pay_days= round((pay  or 400) / (revenue or 5000) * 365, 0),
        )
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# ── CHARTS ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

_DARK = dict(plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
             font=dict(color="#94a3b8", size=11))
_GRID = dict(gridcolor="#1a1a1a", zerolinecolor="#222")
_MARG = dict(l=8, r=8, t=36, b=8)
GREEN = "#22c55e"; BLUE = "#3b82f6"; AMBER = "#f59e0b"; RED = "#ef4444"
PURPLE = "#8b5cf6"; TEAL = "#14b8a6"

def fig_revenue_ebitda(m: dict) -> go.Figure:
    yrs = [f"Y{i+1}" for i in range(m["years"])]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="Revenue", x=yrs, y=m["IS"]["revenue"],
                         marker_color="#1e3a5f", marker_line_width=0), secondary_y=False)
    fig.add_trace(go.Bar(name="EBITDA",  x=yrs, y=m["IS"]["ebitda"],
                         marker_color=GREEN, marker_line_width=0), secondary_y=False)
    fig.add_trace(go.Scatter(name="EBITDA %", x=yrs, y=m["IS"]["ebitda_margin"],
                             mode="lines+markers", line=dict(color=AMBER, width=2),
                             marker=dict(size=6)), secondary_y=True)
    fig.update_layout(barmode="group", title="Revenue & EBITDA (₹ Cr)",
                      xaxis=dict(**_GRID),
                      yaxis=dict(title="₹ Cr", **_GRID),
                      yaxis2=dict(title="Margin %", **_GRID),
                      legend=dict(bgcolor="#111", bordercolor="#222"),
                      margin=_MARG, height=300, **_DARK)
    return fig

def fig_waterfall_ni(m: dict) -> go.Figure:
    yrs = [f"Y{i+1}" for i in range(m["years"])]
    fig = go.Figure()
    for name, data, color in [
        ("Revenue",   m["IS"]["revenue"],  BLUE),
        ("Gross P",   m["IS"]["gross_profit"], TEAL),
        ("EBITDA",    m["IS"]["ebitda"],    GREEN),
        ("EBIT",      m["IS"]["ebit"],      AMBER),
        ("Net Income",m["IS"]["ni"],        PURPLE),
    ]:
        fig.add_trace(go.Scatter(name=name, x=yrs, y=data,
                                 mode="lines+markers", line=dict(width=2),
                                 marker=dict(size=5, color=color),
                                 line_color=color))
    fig.update_layout(title="P&L funnel (₹ Cr)",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      legend=dict(bgcolor="#111", bordercolor="#222"),
                      margin=_MARG, height=300, **_DARK)
    return fig

def fig_bs_composition(m: dict) -> go.Figure:
    yrs = [f"Y{i+1}" for i in range(m["years"])]
    fig = go.Figure()
    for name, data, color in [
        ("Cash",         m["BS"]["cash"],         GREEN),
        ("Receivables",  m["BS"]["receivables"],   BLUE),
        ("Inventory",    m["BS"]["inventory"],     AMBER),
        ("PP&E",         m["BS"]["ppe"],           PURPLE),
        ("Other assets", m["BS"]["other_assets"],  "#475569"),
    ]:
        fig.add_trace(go.Bar(name=name, x=yrs, y=data,
                             marker_color=color, marker_line_width=0))
    fig.update_layout(barmode="stack", title="Asset composition (₹ Cr)",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      legend=dict(bgcolor="#111", bordercolor="#222"),
                      margin=_MARG, height=300, **_DARK)
    return fig

def fig_cfo_fcf(m: dict) -> go.Figure:
    yrs = [f"Y{i+1}" for i in range(m["years"])]
    fcf = [m["IS"]["ni"][i] + m["IS"]["da"][i] + m["CF"]["capex"][i] for i in range(m["years"])]
    colors_cfo = [GREEN if v >= 0 else RED for v in m["CF"]["cfo"]]
    colors_fcf = [TEAL  if v >= 0 else RED for v in fcf]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="CFO",  x=yrs, y=m["CF"]["cfo"], marker_color=colors_cfo, marker_line_width=0))
    fig.add_trace(go.Bar(name="FCF",  x=yrs, y=fcf,            marker_color=colors_fcf, marker_line_width=0, opacity=0.7))
    fig.update_layout(barmode="group", title="Operating cash flow vs Free cash flow (₹ Cr)",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      legend=dict(bgcolor="#111", bordercolor="#222"),
                      margin=_MARG, height=280, **_DARK)
    return fig

def fig_bs_check(m: dict) -> go.Figure:
    yrs = [f"Y{i+1}" for i in range(m["years"])]
    checks = m["BS"]["check"]
    colors = [GREEN if abs(v) < 1 else RED for v in checks]
    fig = go.Figure(go.Bar(x=yrs, y=checks, marker_color=colors,
                           marker_line_width=0,
                           text=[f"₹{v:+.1f}" for v in checks],
                           textposition="outside", textfont_size=10))
    fig.add_hline(y=0, line_color="rgba(255, 255, 255, 0.19)", line_width=1)
    fig.update_layout(title="Balance sheet check: Assets − (Liabilities + Equity)",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      margin=_MARG, height=220, **_DARK)
    return fig

def fig_margins(m: dict) -> go.Figure:
    yrs = [f"Y{i+1}" for i in range(m["years"])]
    fig = go.Figure()
    for name, data, color in [
        ("Gross margin %",  m["IS"]["gross_margin"],  BLUE),
        ("EBITDA margin %", m["IS"]["ebitda_margin"], GREEN),
        ("NI margin %",     m["IS"]["ni_margin"],     AMBER),
    ]:
        fig.add_trace(go.Scatter(name=name, x=yrs, y=data,
                                 mode="lines+markers+text",
                                 text=[f"{v:.1f}%" for v in data],
                                 textposition="top center", textfont_size=9,
                                 line=dict(color=color, width=2),
                                 marker=dict(size=6, color=color)))
    fig.update_layout(title="Margin expansion story (%)",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      legend=dict(bgcolor="#111", bordercolor="#222"),
                      margin=_MARG, height=280, **_DARK)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ── EXPLANATION SYSTEM ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _exp(title, what, why, formula, bench=""):
    b = f"<li><b>Benchmark:</b> {bench}</li>" if bench else ""
    st.markdown(f"""
    <div class="explain-box">
      <b>What is {title}?</b><br>{what}<br><br>
      <b>Why it matters:</b><br>{why}<br><br>
      <b>Formula:</b> <span class="calc">{formula}</span>
      <ul>{b}</ul>
    </div>""", unsafe_allow_html=True)

EXPS = {
    "rev_growth":     ("Revenue CAGR",
        "Compound Annual Growth Rate of revenue — the single most important projection assumption.",
        "Every projection downstream multiplies from revenue. A 1pp change in CAGR over 5 years changes enterprise value by 8–15% depending on the sector.",
        "CAGR = (Exit Revenue / Base Revenue)^(1/n) − 1",
        "IT: 8–15%. FMCG: 8–14%. Auto: 6–12%. Banking: 12–18%."),
    "gross_margin":   ("Gross Margin",
        "Revenue minus Cost of Goods Sold, as a % of Revenue. The most fundamental profitability measure — what's left after making the product.",
        "Gross margin determines how much is available for SGA, R&D, D&A, and ultimately profit. Expanding gross margins signal pricing power or cost efficiency.",
        "Gross Margin % = (Revenue − COGS) / Revenue × 100",
        "IT services: 28–35%. FMCG: 40–55%. Manufacturing: 15–30%. Banking: 65–80%."),
    "ebitda_margin":  ("EBITDA Margin",
        "EBITDA (Earnings Before Interest, Tax, Depreciation, Amortisation) as % of Revenue. The proxy for operating cash generation.",
        "Every M&A valuation, LBO model, and debt covenant uses EBITDA. It strips out financing choices (interest) and accounting policies (D&A), making companies comparable.",
        "EBITDA Margin = EBITDA / Revenue × 100 = Gross Margin − SGA/Rev × 100",
        "IT: 22–28%. FMCG: 18–25%. Paints: 17–22%. Auto: 8–14%."),
    "da_pct":         ("D&A as % of Revenue",
        "Depreciation & Amortisation — the annual non-cash write-down of PP&E and intangible assets.",
        "D&A is added back in the CFS (non-cash) but reduces EBIT and therefore taxes. Asset-heavy industries have higher D&A, which paradoxically means more cash generation relative to reported profit.",
        "D&A % = (Depreciation + Amortisation) / Revenue × 100",
        "IT (asset-light): 1.5–3%. Manufacturing: 4–8%. Telecom: 15–22%."),
    "capex_pct":      ("CapEx as % of Revenue",
        "Capital Expenditure — cash spent acquiring or maintaining long-term physical assets. Classified under Investing Activities in the CFS.",
        "High CapEx businesses generate less Free Cash Flow. Maintenance CapEx (to sustain current capacity) ≈ D&A for mature businesses. Growth CapEx > D&A signals capacity expansion.",
        "CapEx % = Capital Expenditure / Revenue × 100. FCF = EBITDA − Tax − CapEx − ΔNWC",
        "IT: 2–4%. FMCG: 3–6%. Auto: 5–8%. Utilities/Telecom: 15–25%."),
    "receivables_days": ("Receivables Days (DSO)",
        "Days Sales Outstanding — how many days on average it takes the company to collect payment after a sale is made.",
        "An increasing DSO signals customers are paying slower — either a credit risk or worsening bargaining position. DSO feeds directly into Working Capital and therefore FCF.",
        "DSO = (Accounts Receivable / Revenue) × 365",
        "IT (project billing): 60–80 days. FMCG (distributor terms): 20–35. Auto (dealer channel): 5–15."),
    "inventory_days": ("Inventory Days (DIO)",
        "Days Inventory Outstanding — how long products sit in inventory before being sold.",
        "High DIO ties up cash and signals either slow-moving products or supply chain inefficiency. Low DIO = lean operations. Negative working capital businesses (like FMCG) have very low DIO.",
        "DIO = (Inventory / COGS) × 365",
        "FMCG: 30–60 days. Auto: 15–25. Pharma: 90–120. IT (services): near 0."),
    "payables_days":  ("Payables Days (DPO)",
        "Days Payable Outstanding — how long the company takes to pay its suppliers. Higher DPO = more free financing from suppliers.",
        "DPO is a source of free working capital. Large companies (HUL, Maruti) have strong enough bargaining power to extend DPO, improving their cash cycle. DPO > DIO + DSO = negative working capital (cash-generative).",
        "DPO = (Accounts Payable / COGS) × 365",
        "Large FMCG: 50–90 days. Auto: 30–50. IT: 20–40."),
    "dividend_payout": ("Dividend Payout Ratio",
        "Percentage of Net Income distributed to shareholders as dividends. Retained earnings = NI × (1 − payout ratio).",
        "Retained earnings build equity and fund growth without dilution. High-payout companies return cash but grow more slowly. In the model, retained earnings plug into the BS equity balance each year.",
        "Payout ratio = Dividends / Net Income × 100. Retained earnings += NI × (1 − payout)",
        "IT (capital-light): 50–70%. Auto: 25–40%. Banking: 15–25%. High-growth: 0%."),
}


# ═══════════════════════════════════════════════════════════════════════════
# ── PROJECTION ASSUMPTIONS PANEL ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def proj_panel(d: dict, kp: str, years: int) -> dict:
    show_exp = st.toggle("Show detailed explanations for every input", value=False, key=f"{kp}_exp")

    def _maybe(key):
        if show_exp and key in EXPS:
            e = EXPS[key]
            _exp(e[0], e[1], e[2], e[3], e[4] if len(e) > 4 else "")

    with st.expander("⚙️  Projection assumptions", expanded=True):
        st.markdown('<p class="sec">P&L assumptions</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        rev_g = c1.number_input("Revenue CAGR (%)", value=float(d["rev_growth"]),
                                 min_value=0.0, max_value=50.0, step=0.5, key=f"{kp}_rg")
        gm_e  = c2.number_input("Gross margin exit (%)", value=float(d["gross_margin_exit"]),
                                  min_value=0.0, max_value=90.0, step=0.5, key=f"{kp}_gme")
        em_e  = c3.number_input("EBITDA margin exit (%)", value=float(d["ebitda_margin_exit"]),
                                  min_value=0.0, max_value=60.0, step=0.5, key=f"{kp}_eme")
        _maybe("rev_growth"); _maybe("gross_margin"); _maybe("ebitda_margin")

        c1, c2, c3 = st.columns(3)
        da_e  = c1.number_input("D&A exit (% rev)", value=float(d["da_pct_exit"]),
                                  min_value=0.0, max_value=20.0, step=0.25, key=f"{kp}_dae")
        cx    = c2.number_input("CapEx (% rev)", value=float(d["capex_pct"]),
                                  min_value=0.0, max_value=30.0, step=0.5, key=f"{kp}_cx")
        div   = c3.number_input("Dividend payout (%)", value=float(d["dividend_payout"]),
                                  min_value=0.0, max_value=100.0, step=5.0, key=f"{kp}_div")
        _maybe("da_pct"); _maybe("capex_pct"); _maybe("dividend_payout")

        st.markdown('<p class="sec">Working capital (days)</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        recv_d = c1.number_input("Receivables days (DSO)", value=float(d["b_receivables_days"]),
                                   min_value=0.0, max_value=180.0, step=1.0, key=f"{kp}_rd")
        inv_d  = c2.number_input("Inventory days (DIO)", value=float(d["b_inventory_days"]),
                                   min_value=0.0, max_value=365.0, step=1.0, key=f"{kp}_id")
        pay_d  = c3.number_input("Payables days (DPO)", value=float(d["b_payables_days"]),
                                   min_value=0.0, max_value=180.0, step=1.0, key=f"{kp}_pd")
        _maybe("receivables_days"); _maybe("inventory_days"); _maybe("payables_days")

        st.markdown('<p class="sec">Debt & financing</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        dr    = c1.number_input("Annual debt repayment (₹ Cr)", value=float(d["debt_repayment_pa"]),
                                  min_value=0.0, step=100.0, key=f"{kp}_dr")
        nd    = c2.number_input("New debt raised / yr (₹ Cr)", value=float(d.get("new_debt_raised",0)),
                                  min_value=0.0, step=100.0, key=f"{kp}_nd")
        ir    = c3.number_input("Interest rate (%)", value=float(d["interest_rate"]),
                                  min_value=0.0, max_value=25.0, step=0.25, key=f"{kp}_ir")
        tx    = c1.number_input("Tax rate (%)", value=float(d["tax_rate"]),
                                  min_value=0.0, max_value=40.0, step=0.5, key=f"{kp}_tx")

    return dict(
        rev_growth=rev_g, gross_margin_exit=gm_e, ebitda_margin_exit=em_e,
        da_pct_exit=da_e, capex_pct=cx, dividend_payout=div,
        b_receivables_days=recv_d, b_inventory_days=inv_d, b_payables_days=pay_d,
        debt_repayment_pa=dr, new_debt_raised=nd, interest_rate=ir, tax_rate=tx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── RESULTS RENDERER ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _mc(col, lbl, val, sub="", color="#f1f5f9"):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="m-label">{lbl}</div>'
        f'<div class="m-value" style="color:{color}">{val}</div>'
        f'<div class="m-sub">{sub}</div>'
        f'</div>', unsafe_allow_html=True)


def render_results(m: dict, p: dict, name: str):
    IS, BS, CF = m["IS"], m["BS"], m["CF"]
    yrs  = m["years"]
    ylabs = [f"Y{i+1}" for i in range(yrs)]
    base = ["Base"] + ylabs

    # ── Headline KPIs (Y5) ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"## Results — {name}")

    max_check = max(abs(v) for v in BS["check"])
    bal_ok    = max_check < 1.0

    if bal_ok:
        st.markdown(f'<div class="bal-ok">✓ Balance sheet balances — Assets = Liabilities + Equity every year (max imbalance: ₹{max_check:.2f} Cr)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bal-err">✗ Balance sheet imbalance detected — max: ₹{max_check:.1f} Cr. Check working capital assumptions.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(6)
    _mc(cols[0], "Y5 Revenue",    f"₹{IS['revenue'][-1]:,.0f} Cr",  f"{p['rev_growth']:.1f}% CAGR")
    _mc(cols[1], "Y5 EBITDA",     f"₹{IS['ebitda'][-1]:,.0f} Cr",   f"{IS['ebitda_margin'][-1]:.1f}% margin")
    _mc(cols[2], "Y5 Net Income", f"₹{IS['ni'][-1]:,.0f} Cr",       f"{IS['ni_margin'][-1]:.1f}% margin",
        GREEN if IS["ni"][-1] > 0 else RED)
    _mc(cols[3], "Y5 FCF",
        f"₹{IS['ni'][-1]+IS['da'][-1]+CF['capex'][-1]:,.0f} Cr", "NI + D&A − CapEx")
    _mc(cols[4], "Y5 Total Assets", f"₹{BS['total_assets'][-1]:,.0f} Cr", "")
    _mc(cols[5], "Y5 Equity",       f"₹{BS['equity'][-1]:,.0f} Cr",
        f"ROE {IS['ni'][-1]/BS['equity'][-1]*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs(["📈 Income Statement", "🏗️ Balance Sheet", "💵 Cash Flow",
                    "📊 Charts", "📐 Ratio Analysis", "⚙️ BS Check", "📖 Summary"])

    # ── Tab 0: IS ──────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        <div class="chart-exp">
        <b>How to read this income statement:</b> Numbers flow top-down, each line subtracting
        a cost from the prior line. Revenue → subtract COGS → Gross Profit → subtract SGA
        → EBITDA → subtract D&A → EBIT → subtract Interest → EBT → subtract Tax → Net Income.
        The margin % columns show each profit line as a fraction of revenue so you can track
        expansion or compression over time. This is the P&L that feeds directly into the CFS
        (via Net Income at the top) and the BS (via Retained Earnings).
        </div>
        """, unsafe_allow_html=True)

        b_rev = p["b_revenue"]
        b_gm  = p["b_gross_margin"]
        b_em  = p["b_ebitda_margin"]
        b_da  = p["b_da_pct"]

        is_df = pd.DataFrame({
            "Line item": [
                "Revenue (₹ Cr)", "  Revenue growth",
                "COGS (₹ Cr)", "Gross profit (₹ Cr)", "  Gross margin %",
                "Operating expenses (₹ Cr)",
                "EBITDA (₹ Cr)", "  EBITDA margin %",
                "D&A (₹ Cr)", "EBIT (₹ Cr)", "  EBIT margin %",
                "Interest expense (₹ Cr)", "EBT (₹ Cr)",
                "Tax (₹ Cr)", "Net income (₹ Cr)", "  Net margin %",
            ],
            "Base": [
                f"₹{b_rev:,.0f}", "—",
                f"₹{b_rev*(1-b_gm/100):,.0f}",
                f"₹{b_rev*b_gm/100:,.0f}", f"{b_gm:.1f}%",
                f"₹{b_rev*(b_gm/100 - b_em/100):,.0f}",
                f"₹{b_rev*b_em/100:,.0f}", f"{b_em:.1f}%",
                f"₹{b_rev*b_da/100:,.0f}",
                f"₹{b_rev*(b_em-b_da)/100:,.0f}", f"{b_em-b_da:.1f}%",
                "—", "—", "—", "—", "—",
            ],
        })

        for i, yr in enumerate(ylabs):
            rev  = IS["revenue"][i]; gm = IS["gross_margin"][i]; em = IS["ebitda_margin"][i]
            growth = (rev / (IS["revenue"][i-1] if i > 0 else p["b_revenue"]) - 1) * 100
            is_df[yr] = [
                f"₹{rev:,.0f}", f"{growth:.1f}%",
                f"₹{IS['cogs'][i]:,.0f}",
                f"₹{IS['gross_profit'][i]:,.0f}", f"{gm:.1f}%",
                f"₹{IS['opex'][i]:,.0f}",
                f"₹{IS['ebitda'][i]:,.0f}", f"{em:.1f}%",
                f"₹{IS['da'][i]:,.0f}",
                f"₹{IS['ebit'][i]:,.0f}", f"{IS['ebit_margin'][i]:.1f}%",
                f"₹{IS['interest'][i]:,.0f}",
                f"₹{IS['ebt'][i]:,.0f}",
                f"₹{IS['tax'][i]:,.0f}",
                f"₹{IS['ni'][i]:,.0f}", f"{IS['ni_margin'][i]:.1f}%",
            ]

        st.dataframe(is_df, hide_index=True, use_container_width=True)

    # ── Tab 1: BS ──────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
        <div class="chart-exp">
        <b>How to read the balance sheet:</b> Assets (what the company owns) must always equal
        Liabilities + Equity (how those assets are financed). Cash is the plug — it is the ending
        balance from the CFS. PP&E grows with CapEx and shrinks with D&A. Receivables and
        inventory grow with revenue. Equity grows with retained earnings (NI minus dividends).
        If Assets ≠ L+E at any year, the model has an error. The "BS check" tab shows the
        imbalance directly.
        </div>
        """, unsafe_allow_html=True)

        bs_df = pd.DataFrame({"Line item": [
            "ASSETS", "Cash (₹ Cr)", "Accounts receivable (₹ Cr)", "Inventory (₹ Cr)",
            "PP&E net (₹ Cr)", "Other assets (₹ Cr)", "Total assets (₹ Cr)",
            "LIABILITIES & EQUITY",
            "Accounts payable (₹ Cr)", "Short-term debt (₹ Cr)",
            "Long-term debt (₹ Cr)", "Other liabilities (₹ Cr)", "Total liabilities (₹ Cr)",
            "Retained earnings (₹ Cr)", "Total equity (₹ Cr)",
            "Total L+E (₹ Cr)", "BS check (₹ Cr)",
        ]})

        for i, yr in enumerate(ylabs):
            bs_df[yr] = [
                "", f"₹{BS['cash'][i]:,.0f}", f"₹{BS['receivables'][i]:,.0f}",
                f"₹{BS['inventory'][i]:,.0f}", f"₹{BS['ppe'][i]:,.0f}",
                f"₹{BS['other_assets'][i]:,.0f}", f"₹{BS['total_assets'][i]:,.0f}",
                "",
                f"₹{BS['payables'][i]:,.0f}", f"₹{BS['short_debt'][i]:,.0f}",
                f"₹{BS['long_debt'][i]:,.0f}", f"₹{BS['other_liabilities'][i]:,.0f}",
                f"₹{BS['total_liabilities'][i]:,.0f}",
                f"₹{BS['retained_earnings'][i]:,.0f}", f"₹{BS['equity'][i]:,.0f}",
                f"₹{BS['total_le'][i]:,.0f}",
                f"₹{BS['check'][i]:+.1f}",
            ]

        st.dataframe(bs_df, hide_index=True, use_container_width=True)

    # ── Tab 2: CFS ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
        <div class="chart-exp">
        <b>How to read the cash flow statement:</b> The CFS reconciles net income (accounting profit)
        to actual cash. There are three sections:<br>
        <b>Operating (CFO):</b> Start with NI, add back D&A (non-cash), adjust for working capital
        changes (↑receivables = cash outflow, ↑payables = cash inflow). CFO shows how much cash the
        core business generates.<br>
        <b>Investing (CFI):</b> Primarily CapEx (negative = cash spent). A company growing fast has
        large negative CFI.<br>
        <b>Financing (CFF):</b> Debt raised/repaid + dividends paid. The residual sum of CFO+CFI+CFF
        = change in cash, which plugs into the Balance Sheet's cash line. This is the most critical
        linkage in the model.
        </div>
        """, unsafe_allow_html=True)

        cf_df = pd.DataFrame({"Line item": [
            "OPERATING ACTIVITIES",
            "Net income (₹ Cr)", "+ D&A (₹ Cr)", "± Working capital (₹ Cr)",
            "= CFO (₹ Cr)",
            "INVESTING ACTIVITIES",
            "CapEx (₹ Cr)", "= CFI (₹ Cr)",
            "FINANCING ACTIVITIES",
            "Debt raised (₹ Cr)", "Debt repaid (₹ Cr)", "Dividends (₹ Cr)",
            "= CFF (₹ Cr)",
            "Net cash change (₹ Cr)",
            "Opening cash (₹ Cr)", "Closing cash (₹ Cr)",
        ]})

        for i, yr in enumerate(ylabs):
            cf_df[yr] = [
                "",
                f"₹{CF['ni'][i]:,.0f}", f"₹{CF['da'][i]:,.0f}",
                f"₹{CF['delta_wc'][i]:+,.0f}", f"₹{CF['cfo'][i]:,.0f}",
                "",
                f"₹{CF['capex'][i]:,.0f}", f"₹{CF['cfi'][i]:,.0f}",
                "",
                f"₹{CF['debt_raised'][i]:,.0f}", f"₹{CF['debt_repaid'][i]:,.0f}",
                f"₹{CF['dividends'][i]:,.0f}", f"₹{CF['cff'][i]:,.0f}",
                f"₹{CF['net_change'][i]:+,.0f}",
                f"₹{CF['beg_cash'][i]:,.0f}", f"₹{CF['end_cash'][i]:,.0f}",
            ]

        st.dataframe(cf_df, hide_index=True, use_container_width=True)

    # ── Tab 3: Charts ──────────────────────────────────────────────────────
    with tabs[3]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_revenue_ebitda(m), use_container_width=True)
        c1.markdown('<div class="chart-exp">Revenue (blue) and EBITDA (green) growing together — the operating thesis. Yellow line = EBITDA margin % (right axis). Watch for margin expansion: higher margin on growing revenue = compounding EBITDA growth.</div>', unsafe_allow_html=True)
        c2.plotly_chart(fig_waterfall_ni(m),  use_container_width=True)
        c2.markdown('<div class="chart-exp">P&L funnel — five lines, each stripping away a cost. Gross profit > EBITDA > EBIT > Net Income. The gap between lines represents each cost category. A widening gap signals cost pressure; a narrowing gap signals operational leverage.</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        c3.plotly_chart(fig_bs_composition(m), use_container_width=True)
        c3.markdown('<div class="chart-exp">Asset composition growing over time. A rising PP&E bar = capital-intensive growth. Stable receivables as % of revenue = disciplined collections. Rising cash = strong FCF generation.</div>', unsafe_allow_html=True)
        c4.plotly_chart(fig_cfo_fcf(m),        use_container_width=True)
        c4.markdown('<div class="chart-exp">CFO (operating cash) vs FCF (after CapEx). The gap = CapEx spend. Asset-light companies (IT) have CFO ≈ FCF. Capital-intensive companies (Auto, Telecom) have a large gap. Both should grow over time for a healthy company.</div>', unsafe_allow_html=True)

        st.plotly_chart(fig_margins(m), use_container_width=True)
        st.markdown('<div class="chart-exp">Margin expansion story. All three margins should trend upward — gross margin reflecting pricing power or COGS efficiency, EBITDA margin reflecting operating leverage, NI margin compounding both. If NI margin expands faster than EBITDA margin, interest expense is falling (debt being repaid).</div>', unsafe_allow_html=True)

    # ── Tab 4: Ratios ──────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
        <div class="chart-exp">
        <b>Ratio analysis — what recruiters expect you to know:</b><br>
        <b>Profitability:</b> Gross/EBITDA/NI margins — are they expanding?<br>
        <b>Efficiency:</b> ROE (return on equity), ROA (return on assets) — is management generating returns above cost of capital?<br>
        <b>Leverage:</b> Debt/EBITDA — is the company over-levered? Interest coverage — can it service debt?<br>
        <b>Liquidity:</b> Current ratio — can it meet short-term obligations?<br>
        <b>Cash:</b> FCF, cash conversion — how much profit becomes actual cash?
        </div>
        """, unsafe_allow_html=True)

        ratio_df = compute_ratios(m, p)
        st.dataframe(ratio_df, use_container_width=True)

        st.markdown(f"""
        <div class="explain-box">
        <b>Key ratio formulas:</b><br>
        <span class="calc">ROE = Net Income / Equity × 100</span><br>
        <span class="calc">ROA = Net Income / Total Assets × 100</span><br>
        <span class="calc">Interest Coverage = EBITDA / Interest Expense</span><br>
        <span class="calc">Debt/EBITDA = (Short Debt + Long Debt) / EBITDA</span><br>
        <span class="calc">Current Ratio = (Cash + Receivables + Inventory) / (Payables + Short Debt)</span><br>
        <span class="calc">FCF = Net Income + D&A − CapEx</span><br>
        <span class="calc">Cash Conversion = CFO / EBITDA × 100</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 5: BS Check ────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("""
        <div class="chart-exp">
        <b>The balance sheet check is the most important model integrity test.</b><br>
        In a correctly built 3-statement model, Total Assets = Total Liabilities + Equity
        every single year. If this doesn't hold, there is an error somewhere — usually a broken
        linkage between the CFS and BS (ending cash not flowing correctly) or an equity calculation
        error (retained earnings not accumulating properly). The chart below should show bars at or
        near ₹0 Cr. Any non-zero bar is a model bug.
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_bs_check(m), use_container_width=True)

        for i, yr in enumerate(ylabs):
            chk = BS["check"][i]
            if abs(chk) < 1:
                st.markdown(f'<div class="bal-ok">{yr}: ✓ Balanced — Assets ₹{BS["total_assets"][i]:,.0f} Cr = L+E ₹{BS["total_le"][i]:,.0f} Cr (diff: ₹{chk:+.2f})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bal-err">{yr}: ✗ Imbalance ₹{chk:+.1f} Cr — check CFS-BS cash linkage</div>', unsafe_allow_html=True)

    # ── Tab 6: Summary ─────────────────────────────────────────────────────
    with tabs[6]:
        rev_5   = IS["revenue"][-1]
        ni_5    = IS["ni"][-1]
        fcf_5   = ni_5 + IS["da"][-1] + CF["capex"][-1]
        rev_cagr = (rev_5 / p["b_revenue"]) ** (1/yrs) - 1

        st.markdown(f"""
        <div style="background:#0f2318;border:1px solid #238636;border-radius:8px;padding:16px 20px;margin-bottom:14px">
          <div style="font-size:16px;font-weight:700;color:#22c55e">Financial model summary — {name}</div>
          <div style="font-size:12px;color:#94a3b8;margin-top:5px">
            {yrs}-year integrated three-statement projection · Base: FY24 approximates · All figures ₹ Crore
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explain-box">
        <b>What this model projects:</b><br>
        {name} grows revenue from ₹{p['b_revenue']:,.0f} Cr to ₹{rev_5:,.0f} Cr over {yrs} years,
        a CAGR of {rev_cagr*100:.1f}%. EBITDA margins expand from {p['b_ebitda_margin']:.1f}%
        to {IS['ebitda_margin'][-1]:.1f}%, driving EBITDA of ₹{IS['ebitda'][-1]:,.0f} Cr in Year {yrs}.
        Net income reaches ₹{ni_5:,.0f} Cr with a {IS['ni_margin'][-1]:.1f}% net margin.
        Free cash flow of ₹{fcf_5:,.0f} Cr in Year {yrs} supports
        {'continued debt repayment and shareholder returns' if p['debt_repayment_pa'] > 0 else 'reinvestment and growth capital'}.
        The balance sheet
        {'balances correctly every year' if bal_ok else 'has a minor imbalance — check working capital assumptions'}.
        <br><br>
        <b>Three linkages to verify:</b>
        <ul>
        <li><b>NI → CFS:</b> Y1 CFO = ₹{CF['ni'][0]:,.0f} Cr (NI) + ₹{CF['da'][0]:,.0f} Cr (D&A) + ₹{CF['delta_wc'][0]:+,.0f} Cr (WC) = ₹{CF['cfo'][0]:,.0f} Cr</li>
        <li><b>NI → BS:</b> Retained earnings grow by ₹{IS['ni'][0]:,.0f} Cr NI minus ₹{abs(CF['dividends'][0]):,.0f} Cr dividends = ₹{IS['ni'][0]+CF['dividends'][0]:,.0f} Cr added to equity in Y1</li>
        <li><b>CFS → BS:</b> Ending cash Y{yrs} = ₹{CF['end_cash'][-1]:,.0f} Cr = BS cash ₹{BS['cash'][-1]:,.0f} Cr ({'✓ match' if abs(CF['end_cash'][-1] - BS['cash'][-1]) < 2 else '✗ mismatch — debug'})</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Sensitivity: Y5 NI across revenue growth scenarios
        st.markdown("### Quick sensitivity — Y5 Net Income vs Revenue CAGR")
        sens_rows = []
        for test_cagr in [p["rev_growth"]-4, p["rev_growth"]-2, p["rev_growth"],
                          p["rev_growth"]+2, p["rev_growth"]+4]:
            if test_cagr < 0: continue
            pp   = {**p, "rev_growth": max(0, test_cagr)}
            mm   = build_model(pp, yrs)
            sens_rows.append({
                "Revenue CAGR":   f"{test_cagr:.0f}%",
                "Y5 Revenue":     f"₹{mm['IS']['revenue'][-1]:,.0f} Cr",
                "Y5 EBITDA":      f"₹{mm['IS']['ebitda'][-1]:,.0f} Cr",
                "Y5 Net Income":  f"₹{mm['IS']['ni'][-1]:,.0f} Cr",
                "Y5 FCF":         f"₹{mm['IS']['ni'][-1]+mm['IS']['da'][-1]+mm['CF']['capex'][-1]:,.0f} Cr",
                "Note":           "← Base case" if test_cagr == p["rev_growth"] else "",
            })
        st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# ── CUSTOM MODEL BUILDER ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def custom_model():
    st.markdown("## 🔧 Custom Company Model")
    st.markdown("""
    <div class="explain-box">
    Enter an NSE ticker to auto-fetch base-year financials from Yahoo Finance, or fill in manually.
    All values in <b>₹ Crores</b>. The model will build a fully linked 3-statement projection.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    ticker  = c1.text_input("NSE ticker", placeholder="ASIANPAINT.NS", key="cust_t")
    co_name = c2.text_input("Company name", value="My Company", key="cust_n")

    fetched = {}
    if ticker:
        with st.spinner(f"Fetching {ticker}…"):
            fetched = fetch_company_data(ticker)
        if fetched.get("revenue"):
            st.success(f"✓ Fetched: Revenue ₹{fetched['revenue']:,} Cr · EBITDA margin {fetched.get('ebitda_margin','—')}%")
        else:
            st.warning("Could not fetch — enter manually below.")

    st.markdown("### Base-year financials (₹ Cr)")
    ca1, ca2, ca3 = st.columns(3)
    b_rev    = ca1.number_input("Revenue",      value=float(fetched.get("revenue", 5000)),  min_value=1.0, key="cust_rev")
    b_gm     = ca2.number_input("Gross margin (%)", value=float(fetched.get("gross_margin",30.0)), min_value=0.0, max_value=100.0, key="cust_gm")
    b_em     = ca3.number_input("EBITDA margin (%)", value=float(fetched.get("ebitda_margin",15.0)), min_value=0.0, max_value=80.0, key="cust_em")
    cb1, cb2, cb3 = st.columns(3)
    b_da     = cb1.number_input("D&A (% rev)",   value=float(fetched.get("da_pct",3.0)),    min_value=0.0, max_value=20.0, key="cust_da")
    b_cash   = cb2.number_input("Cash",           value=float(fetched.get("cash",500)),      min_value=0.0, key="cust_cash")
    b_ppe    = cb3.number_input("PP&E net",        value=float(fetched.get("ppe",2000)),      min_value=0.0, key="cust_ppe")
    cc1, cc2, cc3 = st.columns(3)
    b_recv   = cc1.number_input("Receivables",    value=float(fetched.get("receivables",500)), min_value=0.0, key="cust_recv")
    b_inv    = cc2.number_input("Inventory",       value=float(fetched.get("inventory",200)),  min_value=0.0, key="cust_inv")
    b_pay    = cc3.number_input("Payables",         value=float(fetched.get("payables",400)),   min_value=0.0, key="cust_pay")
    cd1, cd2, cd3 = st.columns(3)
    b_sdebt  = cd1.number_input("Short-term debt", value=float(fetched.get("short_debt",0)),    min_value=0.0, key="cust_sd")
    b_ldebt  = cd2.number_input("Long-term debt",  value=float(fetched.get("long_debt",1000)),  min_value=0.0, key="cust_ld")
    b_equity = cd3.number_input("Equity",           value=float(fetched.get("equity",3000)),    min_value=1.0, key="cust_eq")

    recv_days = b_recv / b_rev * 365 if b_rev > 0 else 60
    inv_days  = b_inv  / b_rev * 365 if b_rev > 0 else 30
    pay_days  = b_pay  / b_rev * 365 if b_rev > 0 else 35

    n_years = st.selectbox("Projection years", [3, 5, 7], index=1, key="cust_yrs")

    d_cust = dict(
        b_revenue=b_rev, b_gross_margin=b_gm, b_ebitda_margin=b_em,
        b_da_pct=b_da, b_interest_expense=b_ldebt*0.08,
        b_cash=b_cash, b_receivables_days=recv_days,
        b_inventory_days=inv_days, b_payables_days=pay_days,
        b_ppe_net=b_ppe, b_other_assets=b_recv+b_inv,
        b_short_debt=b_sdebt, b_long_debt=b_ldebt,
        b_other_liabilities=b_pay*2, b_equity=b_equity,
        rev_growth=12.0, gross_margin_exit=b_gm+2, ebitda_margin_exit=b_em+2,
        da_pct_exit=b_da, capex_pct=float(fetched.get("capex_pct", 4.0)),
        debt_repayment_pa=b_ldebt*0.1, new_debt_raised=0.0,
        interest_rate=8.5, tax_rate=25.17, dividend_payout=30.0,
    )

    st.markdown("### Projection assumptions")
    ov = proj_panel(d_cust, "cust", n_years)
    full_p = {**d_cust, **ov}

    if st.button("📊 Build 3-Statement Model", type="primary",
                 use_container_width=True, key="run_cust"):
        with st.spinner("Building model…"):
            m = build_model(full_p, n_years)
        st.session_state["cust_m"]  = m
        st.session_state["cust_p"]  = full_p
        st.session_state["cust_nm"] = co_name

    if "cust_m" in st.session_state:
        render_results(st.session_state["cust_m"],
                       st.session_state["cust_p"],
                       st.session_state["cust_nm"])


# ═══════════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div style="background:#0c0c0c;border:1px solid #1a1a1a;border-radius:10px;
                padding:18px 22px;margin-bottom:18px">
      <div style="font-size:10px;color:#475569;letter-spacing:.1em;text-transform:uppercase">
        Portfolio Project · Samaksh Sha · Finance & Economics, FLAME University
      </div>
      <div style="font-size:24px;font-weight:700;color:#f1f5f9;margin-top:3px">
        📊 Three-Statement Integrated Financial Model
      </div>
      <div style="font-size:12px;color:#64748b;margin-top:3px">
        Income Statement → Balance Sheet → Cash Flow Statement · Fully linked · Balance check enforced ·
        5 pre-fitted companies + custom builder with yfinance
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Select company")
        options = list(COMPANIES.keys()) + ["──────────────", "🔧 Custom Model"]
        choice  = st.radio("Company", options,
                           format_func=lambda x: f"{COMPANIES[x]['flag']}  {x}"
                                                  if x in COMPANIES else x,
                           label_visibility="collapsed", key="co_sel")
        n_years = st.selectbox("Projection years", [3, 5, 7], index=1, key="n_yrs")

    if choice in ("──────────────", ""):
        st.info("Select a company from the sidebar.")
        return

    if choice == "🔧 Custom Model":
        custom_model()
        return

    co = COMPANIES[choice]

    st.markdown(f"## {co['flag']}  {choice}")
    c1, c2 = st.columns([3, 1])
    with c1:
        tags = "".join(f'<span class="tag" style="background:#1e3a5f;color:#93c5fd">{t}</span>' for t in co["tags"])
        st.markdown(
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:8px;padding:12px 15px">'
            f'<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">{co["sector"]}</div>'
            f'<div style="font-size:13px;color:#e2e8f0">{co["desc"]}</div>'
            f'<div style="background:#111;border-left:3px solid #3b82f6;border-radius:0 6px 6px 0;'
            f'padding:8px 12px;font-size:12px;color:#94a3b8;margin:8px 0">'
            f'<b style="color:#93c5fd;font-size:10px">Analyst thesis:</b><br>{co["thesis"]}</div>'
            f'{tags}</div>', unsafe_allow_html=True)

    with c2:
        live = {}
        try:
            with st.spinner("Fetching live price…"):
                info  = yf.Ticker(co["ticker"]).info
                price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                mktcap = (info.get("marketCap") or 0) / 1e7
                live  = {"price": price, "mktcap": round(mktcap, 0)}
        except Exception:
            pass
        st.markdown(
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:8px;padding:12px 15px">'
            f'<div style="font-size:10px;color:#64748b;letter-spacing:.07em">NSE · LIVE</div>'
            f'<div style="font-size:22px;font-weight:700;color:#f1f5f9;margin-top:6px">'
            f'₹{live.get("price","—"):,.2f}</div>'
            f'<div style="font-size:11px;color:#64748b;margin-top:4px">'
            f'Mkt cap: ₹{live.get("mktcap","—"):,} Cr<br>'
            f'Base rev: ₹{co["b_revenue"]:,.0f} Cr</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ov = proj_panel(co, kp=choice.replace(" ","_"), years=n_years)
    full_p = {**co, **ov}

    key_m = f"m_{choice}"; key_p = f"p_{choice}"

    if st.button(f"📊  Build Model — {choice}", type="primary",
                 use_container_width=True, key=f"run_{choice.replace(' ','_')}"):
        with st.spinner("Building three-statement model…"):
            m = build_model(full_p, n_years)
        st.session_state[key_m] = m
        st.session_state[key_p] = full_p

    if key_m in st.session_state:
        render_results(st.session_state[key_m], st.session_state[key_p], choice)


if __name__ == "__main__":
    main()