import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pycountry
import joblib
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════
# ⚙️  CONFIG — CHANGE EVERYTHING HERE, NOTHING ELSE NEEDS TOUCHING
# ══════════════════════════════════════════════════════════════════
DATA_FILE="./hist_and_pred_df_third_attempt.csv"
MODEL_PATH="./final_regressor.pkl"


COL_PCT_WOMEN   = "pct_women_chamber"
COL_TOTAL_WOMEN = "total_women"
COL_YEAR        = "Year"
COL_ISO3        = "ISO3"
COL_REGION      = "Region"
COL_IS_FUTURE   = "is_future"

COL_PRED_TOTAL     = "pred_total_women"
COL_PRED_LOWER     = "pred_lower"
COL_PRED_UPPER     = "pred_upper"
COL_PRED_CHANGE    = "predicted_change"
COL_PRED_DIRECTION = "pred_direction_label"
COL_LAST_KNOWN     = "last_known_women"

# number of future election cycles to show in charts
N_FUTURE_CYCLES = 3

EXOGENOUS_FEATURES = [
    "hdi", "gdp_growth", "gii", "legislative_size_direct",
    "rdi_score", "effective.quota", "internet_users",
]
LAG_FEATURES = [
    "change_pct_women_lag1", "change_pct_women_lag2",
    "total_women_lag1", "total_women_lag2",
    "chang_pct_women_momentum", "total_women_momentum",
]
MODEL_FEATURES = EXOGENOUS_FEATURES + LAG_FEATURES

FEATURE_LABELS = {
    "hdi":                     "Human Development Index",
    "gdp_growth":              "GDP Growth Rate",
    "gii":                     "Gender Inequality Index",
    "legislative_size_direct": "Parliament Size",
    "rdi_score":               "Democracy Index",
    "effective.quota":         "Gender Quota Strength",
    "internet_users":          "Internet Access",
}

FEATURE_INFO = {
    "hdi": {
        "label": "Human Development Index (HDI)",
        "desc":  "A score from 0 to 1 measuring a country's overall development — combining life expectancy, education, and income. Higher = more developed.",
        "col": "hdi", "icon": "🌱",
        "min": 0.2, "max": 1.0, "step": 0.01, "default": 0.7,
    },
    "gdp_growth": {
        "label": "GDP Growth Rate (%)",
        "desc":  "How fast the economy is growing year-on-year. Positive = growth, negative = recession.",
        "col": "gdp_growth", "icon": "📈",
        "min": -10.0, "max": 15.0, "step": 0.1, "default": 2.0,
    },
    "gii": {
        "label": "Gender Inequality Index (GII)",
        "desc":  "A score from 0 to 1 measuring gaps between men and women in health, education, and political power. Lower = more equal.",
        "col": "gii", "icon": "⚖️",
        "min": 0.0, "max": 0.9, "step": 0.01, "default": 0.3,
    },
    "legislative_size_direct": {
        "label": "Size of Parliament (directly elected seats)",
        "desc":  "Total number of seats directly elected by voters.",
        "col": "legislative_size_direct", "icon": "🏛️",
        "min": 20, "max": 700, "step": 5, "default": 200,
    },
    "rdi_score": {
        "label": "Regional Democracy Index (RDI)",
        "desc":  "A score from 0 to 1 measuring the strength of democratic institutions. Higher = stronger democracy.",
        "col": "rdi_score", "icon": "🗳️",
        "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5,
    },
    "effective.quota": {
        "label": "Gender Quota Strength",
        "desc":  "Whether the country legally requires a minimum share of women candidates. 0 = no quota, 1 = weak quota, 2 = strong enforced quota.",
        "col": "effective.quota", "icon": "📋",
        "min": 0, "max": 2, "step": 1, "default": 0,
    },
    "internet_users": {
        "label": "Internet Access (%)",
        "desc":  "Percentage of the population with internet access. Linked to greater civic participation.",
        "col": "internet_users", "icon": "🌐",
        "min": 0.0, "max": 100.0, "step": 0.5, "default": 70.0,
    },
}

GEMINI_MODEL = "gemini-2.5-flash-lite"

# ══════════════════════════════════════════════════════════════════
# END CONFIG
# ══════════════════════════════════════════════════════════════════

REGIONS = {
    "AFG":"Asia","AGO":"Sub-Saharan Africa","ALB":"Europe","ARE":"MENA",
    "ARG":"Americas","ARM":"Asia","ATG":"Americas","AUS":"Pacific",
    "AUT":"Europe","AZE":"Asia","BDI":"Sub-Saharan Africa","BEL":"Europe",
    "BEN":"Sub-Saharan Africa","BFA":"Sub-Saharan Africa","BGD":"Asia",
    "BGR":"Europe","BHR":"MENA","BHS":"Americas","BIH":"Europe",
    "BLR":"Europe","BLZ":"Americas","BOL":"Americas","BRA":"Americas",
    "BRB":"Americas","BRN":"Asia","BTN":"Asia","BWA":"Sub-Saharan Africa",
    "CAF":"Sub-Saharan Africa","CAN":"Americas","CHE":"Europe","CHL":"Americas",
    "CHN":"Asia","CIV":"Sub-Saharan Africa","CMR":"Sub-Saharan Africa",
    "COD":"Sub-Saharan Africa","COG":"Sub-Saharan Africa","COL":"Americas",
    "COM":"Sub-Saharan Africa","CPV":"Sub-Saharan Africa","CRI":"Americas",
    "CUB":"Americas","CYP":"Europe","CZE":"Europe","DEU":"Europe",
    "DJI":"Sub-Saharan Africa","DMA":"Americas","DNK":"Europe","DOM":"Americas",
    "DZA":"MENA","ECU":"Americas","EGY":"MENA","ESP":"Europe","EST":"Europe",
    "ETH":"Sub-Saharan Africa","FIN":"Europe","FJI":"Pacific","FRA":"Europe",
    "FSM":"Pacific","GAB":"Sub-Saharan Africa","GBR":"Europe","GEO":"Europe",
    "GHA":"Sub-Saharan Africa","GIN":"Sub-Saharan Africa","GMB":"Sub-Saharan Africa",
    "GNB":"Sub-Saharan Africa","GNQ":"Sub-Saharan Africa","GRC":"Europe",
    "GRD":"Americas","GTM":"Americas","GUY":"Americas","HND":"Americas",
    "HRV":"Europe","HTI":"Americas","HUN":"Europe","IDN":"Asia","IND":"Asia",
    "IRL":"Europe","IRN":"Asia","IRQ":"MENA","ISL":"Europe","ISR":"Asia",
    "ITA":"Europe","JAM":"Americas","JOR":"MENA","JPN":"Asia","KAZ":"Europe",
    "KEN":"Sub-Saharan Africa","KGZ":"Europe","KHM":"Asia","KIR":"Pacific",
    "KOR":"Asia","KWT":"MENA","LAO":"Asia","LBN":"MENA","LBR":"Sub-Saharan Africa",
    "LBY":"MENA","LCA":"Americas","LIE":"Europe","LKA":"Asia",
    "LSO":"Sub-Saharan Africa","LTU":"Europe","LUX":"Europe","LVA":"Europe",
    "MAR":"MENA","MDA":"Europe","MDG":"Sub-Saharan Africa","MDV":"Asia",
    "MEX":"Americas","MHL":"Pacific","MKD":"Europe","MLI":"Sub-Saharan Africa",
    "MLT":"Europe","MMR":"Asia","MNE":"Europe","MNG":"Asia",
    "MOZ":"Sub-Saharan Africa","MRT":"Sub-Saharan Africa","MUS":"Sub-Saharan Africa",
    "MWI":"Sub-Saharan Africa","MYS":"Asia","NAM":"Sub-Saharan Africa",
    "NER":"Sub-Saharan Africa","NGA":"Sub-Saharan Africa","NIC":"Americas",
    "NLD":"Europe","NOR":"Europe","NPL":"Asia","NRU":"Pacific","NZL":"Pacific",
    "OMN":"MENA","PAK":"Asia","PAN":"Americas","PER":"Americas","PHL":"Asia",
    "PLW":"Pacific","PNG":"Pacific","POL":"Europe","PRK":"Asia","PRT":"Europe",
    "PRY":"Americas","QAT":"MENA","ROU":"Europe","RUS":"Europe",
    "RWA":"Sub-Saharan Africa","SAU":"MENA","SDN":"MENA",
    "SEN":"Sub-Saharan Africa","SGP":"Asia","SLB":"Pacific",
    "SLE":"Sub-Saharan Africa","SLV":"Americas","SMR":"Europe",
    "SOM":"Sub-Saharan Africa","SRB":"Europe","SSD":"Sub-Saharan Africa",
    "STP":"Sub-Saharan Africa","SUR":"Americas","SVK":"Europe","SVN":"Europe",
    "SWE":"Europe","SWZ":"Sub-Saharan Africa","SYC":"Sub-Saharan Africa",
    "SYR":"MENA","TCD":"Sub-Saharan Africa","TGO":"Sub-Saharan Africa",
    "THA":"Asia","TJK":"Europe","TKM":"Europe","TLS":"Asia","TON":"Pacific",
    "TTO":"Americas","TUN":"MENA","TUR":"Europe","TUV":"Pacific",
    "TZA":"Sub-Saharan Africa","UGA":"Sub-Saharan Africa","UKR":"Europe",
    "URY":"Americas","USA":"Americas","UZB":"Europe","VCT":"Americas",
    "VEN":"Americas","VNM":"Asia","VUT":"Pacific","WSM":"Pacific",
    "YEM":"MENA","ZAF":"Sub-Saharan Africa","ZMB":"Sub-Saharan Africa",
    "ZWE":"Sub-Saharan Africa"
}

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Predicting Women's Seats in Parliament",
    page_icon="🌺", layout="wide"
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Playfair+Display:wght@700&display=swap');
html, body, [class*="css"], p, div, span, label, li, h1, h2, h3, h4, h5, h6 {
    font-family: 'Nunito', sans-serif !important; color: #1a1a1a !important;
}
.stApp { background-color: #fffbf2 !important; }
[data-testid="collapsedControl"] { display: none !important; }
button[kind="icon"] { display: none !important; }
[data-testid="stSidebar"] {
    background-color: #fffbf2 !important; background-image: none !important;
    border-right: 3px solid #0d3320 !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
    color: #0d3320 !important; font-family: 'Nunito', sans-serif !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: white !important; color: #0d3320 !important;
    border: 1.5px solid #0d3320 !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder { color: rgba(13,51,32,0.5) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: white !important; border: 1.5px solid #0d3320 !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div > div { color: #0d3320 !important; }
[data-testid="stSidebar"] .stSelectbox svg { fill: #0d3320 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(13,51,32,0.2) !important; }
div[data-testid="stCheckbox"] label p { font-size: 14px !important; color: #0d3320 !important; font-weight: 600 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; border: 2px solid #0d3320 !important;
    color: #0d3320 !important; border-radius: 24px !important; font-weight: 700 !important; font-size: 13px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #e8354a !important; border-color: #e8354a !important; color: white !important;
}
[data-baseweb="popover"], [data-baseweb="popover"] *, [data-baseweb="popover"] ul li,
[data-baseweb="popover"] ul li span, [data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] span {
    color: #1a1a1a !important; font-family: 'Nunito', sans-serif !important;
    font-size: 14px !important; background-color: #fffbf2 !important;
}
[data-baseweb="popover"] [role="option"]:hover { background-color: #f0f5ee !important; }
.stSelectbox > div > div { background: white !important; border: 1.5px solid #ddd !important; border-radius: 8px !important; }
.stSelectbox > div > div > div { color: #1a1a1a !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 4px; border-bottom: 2px solid rgba(13,51,32,0.15) !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Nunito', sans-serif !important; font-weight: 700 !important; font-size: 14px !important; color: #555 !important; padding: 10px 20px !important; border-radius: 10px 10px 0 0 !important; }
.stTabs [aria-selected="true"] { color: #0d3320 !important; border-bottom: 3px solid #0d3320 !important; background: rgba(13,51,32,0.04) !important; }
[data-testid="stSlider"] label p { color: #1a1a1a !important; font-weight: 700 !important; font-size: 14px !important; }
[data-testid="metric-container"] { background: white !important; border-radius: 14px !important; padding: 16px !important; border: 2px solid #f9c74f !important; box-shadow: 0 3px 12px rgba(13,51,32,0.08) !important; }
[data-testid="metric-container"] label { font-size: 13px !important; font-weight: 700 !important; color: #0d3320 !important; }
[data-testid="stMetricValue"] { font-size: 24px !important; color: #e8354a !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; color: #0d3320 !important; font-weight: 600 !important; }
.hoverlayer .hovertext rect { fill: #fffbf2 !important; stroke: #cccccc !important; }
.hoverlayer .hovertext text { fill: #1a1a1a !important; font-family: 'Nunito', sans-serif !important; font-size: 13px !important; }
.js-plotly-plot .plotly .xtick text, .js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .g-xtitle text, .js-plotly-plot .plotly .g-ytitle text,
.js-plotly-plot .plotly .legend text, .js-plotly-plot .plotly .annotation text,
.js-plotly-plot .plotly .cb text { fill: #1a1a1a !important; font-family: 'Nunito', sans-serif !important; font-size: 13px !important; }
.fact-card { background: white; border-radius: 18px; padding: 28px 24px 20px 24px; border-top: 5px solid #e8354a; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 16px; min-height: 170px; position: relative; }
.fact-card.green { border-top-color: #0d3320; } .fact-card.teal { border-top-color: #0097a7; }
.fact-card.gold  { border-top-color: #d4a017; } .fact-card.blue { border-top-color: #1565c0; }
.fact-icon { position: absolute; top: 16px; right: 20px; font-size: 20px; opacity: 0.4; }
.fact-number { font-family: 'Nunito', sans-serif !important; font-size: 56px; font-weight: 800; color: #e8354a; line-height: 1; margin-bottom: 10px; }
.fact-number.green { color: #0d3320; } .fact-number.teal { color: #0097a7; }
.fact-number.gold  { color: #d4a017; } .fact-number.blue { color: #1565c0; }
.fact-label { font-family: 'Nunito', sans-serif; font-size: 14px; color: #1a1a1a; line-height: 1.6; font-weight: 600; }
.fact-sub   { font-family: 'Nunito', sans-serif; font-size: 12px; color: #666; margin-top: 8px; font-style: italic; }
.chat-user { background: #0d3320; border-radius: 16px 16px 4px 16px; padding: 12px 16px; margin: 8px 0 8px auto; max-width: 80%; font-family: 'Nunito', sans-serif; font-size: 14px; font-weight: 600; }
.chat-user, .chat-user * { color: #fdf5e0 !important; }
.chat-ai { background: white; border: 1.5px solid rgba(13,51,32,0.15); border-radius: 16px 16px 16px 4px; padding: 12px 16px; margin: 8px auto 8px 0; max-width: 85%; font-family: 'Nunito', sans-serif; font-size: 14px; line-height: 1.7; }
hr { border-color: rgba(13,51,32,0.12) !important; }
.stAlert > div { background: #e8f5e9 !important; color: #0d3320 !important; border-color: #66bb6a !important; font-family: 'Nunito', sans-serif !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # fix is_future
    df[COL_IS_FUTURE] = df[COL_IS_FUTURE].map(
        {True: True, False: False, "true": True, "false": False,
         "True": True, "False": False, 1: True, 0: False}
    ).fillna(False).astype(bool)

    # keep only first N_FUTURE_CYCLES future rows per country
    hist = df[df[COL_IS_FUTURE] == False]
    fut  = df[df[COL_IS_FUTURE] == True]
    fut  = (fut.sort_values(COL_YEAR)
               .groupby(COL_ISO3)
               .head(N_FUTURE_CYCLES)
               .reset_index(drop=True))
    df = pd.concat([hist, fut], ignore_index=True).sort_values([COL_ISO3, COL_YEAR]).reset_index(drop=True)

    # compute pct_women_chamber for future rows
    if "chamber_total_seats" in df.columns and COL_PRED_TOTAL in df.columns:
        fut_mask = df[COL_IS_FUTURE] == True
        df.loc[fut_mask, COL_PCT_WOMEN] = (
            df.loc[fut_mask, COL_PRED_TOTAL] / df.loc[fut_mask, "chamber_total_seats"] * 100
        ).round(1)

    def iso_to_name(iso):
        try: return pycountry.countries.get(alpha_3=iso).name
        except: return iso
    df["country_name"] = df[COL_ISO3].apply(iso_to_name)
    if "region" not in df.columns:
        df["region"] = df[COL_ISO3].map(REGIONS)
    else:
        df["region"] = df["region"].fillna(df[COL_ISO3].map(REGIONS))
    return df

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Model file found but failed to load: {e}")
            return None
    return None

@st.cache_data
def build_filled_df(_df):
    all_years = list(range(1990, 2051))
    rows = []
    for iso in _df[COL_ISO3].unique():
        cdf = _df[_df[COL_ISO3] == iso].sort_values(COL_YEAR)
        if cdf.empty: continue
        cname  = cdf.iloc[-1]["country_name"]
        region = cdf.iloc[-1]["region"]
        for yr in all_years:
            past = cdf[cdf[COL_YEAR] <= yr]
            if past.empty: continue
            r         = past.iloc[-1]
            raw_fut   = r.get(COL_IS_FUTURE, False)
            is_future = raw_fut == True or str(raw_fut).lower() == "true"
            try:    pct = float(r[COL_PCT_WOMEN])
            except: pct = np.nan
            try:    pred_total = int(r[COL_PRED_TOTAL]) if is_future else np.nan
            except: pred_total = np.nan
            rows.append({
                "ISO3":          iso,
                "Year":          yr,
                "country_name":  cname,
                "region":        region,
                "pct":           pct,
                "pred_total":    pred_total,
                "last_election": int(r[COL_YEAR]),
                "is_predicted":  is_future,
            })
    return pd.DataFrame(rows)

@st.cache_data
def build_chat_context(_df):
    lines = []

    # use the actual Region column from CSV
    REGION_COL = "Region" if "Region" in _df.columns else "region"

    hist   = _df[_df[COL_IS_FUTURE] == False].sort_values(COL_YEAR)
    latest = hist.groupby(COL_ISO3).last().reset_index()

    # ── historical data: most recent per country ──────────────────
    lines.append("HISTORICAL DATA — most recent election per country:")
    for _, row in latest.iterrows():
        try:    pct_str   = f"{float(row[COL_PCT_WOMEN]):.1f}%"
        except: pct_str   = "N/A"
        try:    total_str = f"{int(row[COL_TOTAL_WOMEN])} women"
        except: total_str = ""
        region = str(row.get(REGION_COL, ""))
        cname  = row.get("country_name", row[COL_ISO3])
        lines.append(
            f"  {cname} ({row[COL_ISO3]}) [{region}]: "
            f"{pct_str} ({total_str}) in {int(row[COL_YEAR])}"
        )

    # ── regional averages from historical data ────────────────────
    lines.append("\nREGIONAL AVERAGES — computed from most recent election per country:")
    for reg, grp in latest.groupby(REGION_COL):
        pcts = pd.to_numeric(grp[COL_PCT_WOMEN], errors="coerce").dropna()
        tots = pd.to_numeric(grp[COL_TOTAL_WOMEN], errors="coerce").dropna()
        if len(pcts) > 0:
            lines.append(
                f"  {reg}: avg {pcts.mean():.1f}% "
                f"(range {pcts.min():.1f}%–{pcts.max():.1f}%, "
                f"n={len(pcts)} countries, "
                f"total ~{int(tots.sum())} women)"
            )

    # ── full historical trend per country ─────────────────────────
    lines.append("\nHISTORICAL TREND — all elections per country:")
    for iso, country_hist in hist.groupby(COL_ISO3):
        country_hist = country_hist.sort_values(COL_YEAR)
        cname  = country_hist.iloc[-1].get("country_name", iso)
        region = str(country_hist.iloc[-1].get(REGION_COL, ""))

        # build compact per-election string
        elections = []
        for _, r in country_hist.iterrows():
            try:
                pct = float(r[COL_PCT_WOMEN])
                yr  = int(r[COL_YEAR])
                elections.append(f"{yr}:{pct:.1f}%")
            except:
                continue

        if len(elections) >= 2:
            ep   = float(country_hist.iloc[0][COL_PCT_WOMEN])
            lp   = float(country_hist.iloc[-1][COL_PCT_WOMEN])
            ey   = int(country_hist.iloc[0][COL_YEAR])
            ly   = int(country_hist.iloc[-1][COL_YEAR])
            chg  = lp - ep
            trend = f"{'↑' if chg >= 0 else '↓'}{abs(chg):.1f}pp over {ly - ey} years"
            lines.append(
                f"  {cname} [{region}] ({trend}): {', '.join(elections)}"
            )
        elif len(elections) == 1:
            lines.append(f"  {cname} [{region}]: {elections[0]}")

    # ── predictions ───────────────────────────────────────────────
    future = _df[_df[COL_IS_FUTURE] == True]
    if not future.empty:
        lines.append(f"\nMODEL PREDICTIONS — next {N_FUTURE_CYCLES} elections per country:")
        first_fut = (future.sort_values(COL_YEAR)
                           .groupby(COL_ISO3)
                           .head(N_FUTURE_CYCLES)
                           .reset_index(drop=True))
        for _, row in first_fut.iterrows():
            try:    pct_str   = f"{float(row[COL_PCT_WOMEN]):.1f}%"
            except: pct_str   = "N/A"
            try:    total_str = f"{int(row[COL_PRED_TOTAL])} women"
            except: total_str = ""
            region = str(row.get(REGION_COL, ""))
            cname  = row.get("country_name", row[COL_ISO3])
            lines.append(
                f"  {cname} ({row[COL_ISO3]}) [{region}]: "
                f"predicted {pct_str} ({total_str}) in {int(row[COL_YEAR])}"
            )

        # predicted regional averages
        lines.append("\nPREDICTED REGIONAL AVERAGES — next election:")
        first_only = future.sort_values(COL_YEAR).groupby(COL_ISO3).first().reset_index()
        # use CSV Region column, fall back to REGIONS map
        if REGION_COL in first_only.columns:
            first_only["_reg"] = first_only[REGION_COL]
        else:
            first_only["_reg"] = first_only[COL_ISO3].map(REGIONS)
        for reg, grp in first_only.groupby("_reg"):
            pcts = pd.to_numeric(grp[COL_PCT_WOMEN], errors="coerce").dropna()
            tots = pd.to_numeric(grp[COL_PRED_TOTAL], errors="coerce").dropna()
            if len(pcts) > 0:
                lines.append(
                    f"  {reg}: avg predicted {pcts.mean():.1f}% "
                    f"(range {pcts.min():.1f}%–{pcts.max():.1f}%, "
                    f"n={len(pcts)} countries, "
                    f"total ~{int(tots.sum())} women predicted)"
                )

    return "\n".join(lines)
#@st.cache_data
# def build_chat_context(_df):
#     lines = []
#     hist   = _df[_df[COL_IS_FUTURE] == False].sort_values(COL_YEAR)
#     latest = hist.groupby(COL_ISO3).last().reset_index()

#     lines.append("HISTORICAL DATA — most recent election per country:")
#     for _, row in latest.iterrows():
#         try:    pct_str   = f"{float(row[COL_PCT_WOMEN]):.1f}%"
#         except: pct_str   = "N/A"
#         try:    total_str = f"{int(row[COL_TOTAL_WOMEN])} women"
#         except: total_str = ""
#         region = str(row.get("Region", ""))
#         lines.append(
#             f"  {row.get('country_name', row[COL_ISO3])} ({row[COL_ISO3]}) "
#             f"[{region}]: {pct_str} ({total_str}) in {int(row[COL_YEAR])}"
#         )

#     lines.append("\nREGIONAL AVERAGES — computed from most recent election per country:")
#     for reg, grp in latest.groupby("Region"):
#         pcts = pd.to_numeric(grp[COL_PCT_WOMEN], errors="coerce").dropna()
#         tots = pd.to_numeric(grp[COL_TOTAL_WOMEN], errors="coerce").dropna()
#         if len(pcts) > 0:
#             lines.append(
#                 f"  {reg}: avg {pcts.mean():.1f}% women "
#                 f"(range {pcts.min():.1f}%–{pcts.max():.1f}%, "
#                 f"n={len(pcts)} countries, "
#                 f"total ~{int(tots.sum())} women in parliament)"
#             )

#     lines.append("\nHISTORICAL TREND — earliest to most recent per country:")
#     earliest = hist.groupby(COL_ISO3).first().reset_index()
#     for iso in latest[COL_ISO3]:
#         er = earliest[earliest[COL_ISO3] == iso]
#         lr = latest[latest[COL_ISO3] == iso]
#         if er.empty or lr.empty: continue
#         try:
#             ep  = float(er.iloc[0][COL_PCT_WOMEN])
#             lp  = float(lr.iloc[0][COL_PCT_WOMEN])
#             ey  = int(er.iloc[0][COL_YEAR])
#             ly  = int(lr.iloc[0][COL_YEAR])
#             cn  = lr.iloc[0].get("country_name", iso)
#             reg = str(lr.iloc[0].get("Region", ""))
#             chg = lp - ep
#             lines.append(
#                 f"  {cn} [{reg}]: {ep:.1f}% in {ey} → {lp:.1f}% in {ly} "
#                 f"({'↑' if chg >= 0 else '↓'}{abs(chg):.1f}pp over {ly - ey} years)"
#             )
#         except: continue

#     future = _df[_df[COL_IS_FUTURE] == True]
#     if not future.empty:
#         lines.append(f"\nMODEL PREDICTIONS — next {N_FUTURE_CYCLES} elections per country:")
#         first_fut = (future.sort_values(COL_YEAR)
#                            .groupby(COL_ISO3)
#                            .head(N_FUTURE_CYCLES)
#                            .reset_index(drop=True))
#         for _, row in first_fut.iterrows():
#             try:    pct_str   = f"{float(row[COL_PCT_WOMEN]):.1f}%"
#             except: pct_str   = "N/A"
#             try:    total_str = f"{int(row[COL_PRED_TOTAL])} women"
#             except: total_str = ""
#             region = str(row.get("Region", ""))
#             lines.append(
#                 f"  {row.get('country_name', row[COL_ISO3])} ({row[COL_ISO3]}) "
#                 f"[{region}]: predicted {pct_str} ({total_str}) in {int(row[COL_YEAR])}"
#             )

#         lines.append("\nPREDICTED REGIONAL AVERAGES — next election:")
#         first_only = future.sort_values(COL_YEAR).groupby(COL_ISO3).first().reset_index()
#         first_only["Region"] = first_only[COL_ISO3].map(REGIONS)
#         for reg, grp in first_only.groupby("Region"):
#             pcts = pd.to_numeric(grp[COL_PCT_WOMEN], errors="coerce").dropna()
#             tots = pd.to_numeric(grp[COL_PRED_TOTAL], errors="coerce").dropna()
#             if len(pcts) > 0:
#                 lines.append(
#                     f"  {reg}: avg predicted {pcts.mean():.1f}% "
#                     f"(range {pcts.min():.1f}%–{pcts.max():.1f}%, "
#                     f"n={len(pcts)} countries, "
#                     f"total ~{int(tots.sum())} women predicted)"
#                 )

#     return "\n".join(lines)

df        = load_data()
model     = load_model()
filled_df = build_filled_df(df)
chat_ctx  = build_chat_context(df)

all_countries = sorted(df["country_name"].unique())
all_regions   = ["All regions"] + sorted(df["region"].dropna().unique())

@st.cache_data
def world_avg_by_year(_filled):
    return _filled.groupby("Year")["pct"].mean().reset_index()

world_avg_df = world_avg_by_year(filled_df)

# ══════════════════════════════════════════════════════════════════
# PREDICTION & SHAP
# ══════════════════════════════════════════════════════════════════
def run_prediction(feature_dict):
    if model is None:
        return None
    try:
        row  = pd.DataFrame([{f: feature_dict.get(f, 0) for f in MODEL_FEATURES}])
        pred = float(model.predict(row)[0])
        return {"pct_change": pred, "is_positive": pred >= 0}
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return None

def run_shap(feature_dict):
    if model is None:
        return None
    try:
        import shap
        row       = pd.DataFrame([{f: feature_dict.get(f, 0) for f in MODEL_FEATURES}])
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(row)
        if isinstance(shap_vals, list):
            vals = shap_vals[0][0]
        elif hasattr(shap_vals, "values"):
            vals = shap_vals.values[0]
        else:
            vals = shap_vals[0]
        all_shap = dict(zip(MODEL_FEATURES, vals))
        return {k: all_shap[k] for k in EXOGENOUS_FEATURES if k in all_shap}
    except Exception as e:
        st.warning(f"SHAP error: {e}")
        return None

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
if "selected"      not in st.session_state: st.session_state.selected      = ["Sweden", "Rwanda", "Japan", "United Kingdom"]
if "clear_counter" not in st.session_state: st.session_state.clear_counter = 0
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-family:Nunito,sans-serif;font-size:11px;font-weight:800;"
        "color:rgba(13,51,32,0.45);letter-spacing:2px;text-transform:uppercase;"
        "padding-top:4px;'>Dashboard</div>"
        "<div style='font-family:Playfair Display,serif;font-size:22px;"
        "font-weight:700;color:#f4a7b9;margin:2px 0;'>CinnaRollz</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:12px;"
        "color:rgba(13,51,32,0.55);margin-bottom:12px;'>"
        "🌺 Women's Seats in Parliament</div>",
        unsafe_allow_html=True
    )
    st.divider()
    if st.button("✕ Clear selection", use_container_width=True):
        st.session_state.selected = []
        st.session_state.clear_counter += 1
        st.rerun()
    st.markdown(" ")
    region_filter = st.selectbox("🌍 Filter by region", all_regions)
    search        = st.text_input("🔍 Search a country")
    filtered      = all_countries
    if region_filter != "All regions":
        region_isos = {iso for iso, r in REGIONS.items() if r == region_filter}
        filtered    = [c for c in filtered
                       if not df[df["country_name"] == c].empty
                       and df[df["country_name"] == c][COL_ISO3].iloc[0] in region_isos]
    if search:
        filtered = [c for c in filtered if search.lower() in c.lower()]
    for country in filtered:
        checked = country in st.session_state.selected
        ticked  = st.checkbox(country, value=checked,
                               key=f"cb_{country}_{st.session_state.clear_counter}")
        if ticked and country not in st.session_state.selected:
            st.session_state.selected.append(country)
        elif not ticked and country in st.session_state.selected:
            st.session_state.selected.remove(country)

selected = st.session_state.selected

# ══════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════
COLORS = ["#e8354a","#f4511e","#f9c74f","#0d3320","#0097a7","#43a047","#1565c0","#8e24aa","#fb8c00","#00838f"]
AXIS   = dict(color="#1a1a1a", tickfont=dict(color="#1a1a1a", size=13, family="Nunito, sans-serif"), gridcolor="rgba(13,51,32,0.08)")

def base_layout():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito, sans-serif", color="#1a1a1a", size=14),
        margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified",
        hoverlabel=dict(bgcolor="#fffbf2", bordercolor="#cccccc",
                        font=dict(color="#1a1a1a", size=13, family="Nunito, sans-serif")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color="#1a1a1a", size=13, family="Nunito, sans-serif"))
    )

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:24px 0 10px 0;">
  <div style="font-size:38px; letter-spacing:6px; margin-bottom:10px;">🌺 🥐 🌿</div>
  <div style="font-family:'Nunito',sans-serif; font-size:30px; font-weight:800; color:#1a1a1a; line-height:1.1;">
    Predicting Women's Seats in Parliament
  </div>
  <div style="font-family:'Playfair Display',serif; font-size:19px; font-weight:700; color:#f4a7b9; margin-top:6px;">
    CinnaRollz
  </div>
  <div style="font-family:'Nunito',sans-serif; font-size:12px; color:#888; font-weight:600; letter-spacing:2px; text-transform:uppercase; margin-top:6px;">
    Historical Data & Predictions · 162 Countries · 1990–2040
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_facts, tab_line, tab_bar, tab_map, tab_explorer, tab_insights, tab_chat = st.tabs([
    "🌺  Why it matters", "📈  Trends", "📊  Snapshot",
    "🗺️  Map", "🔍  What drives the prediction?",
    "🌏  Insights", "💬  Ask the data",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — WHY IT MATTERS
# ══════════════════════════════════════════════════════════════════
with tab_facts:
    st.markdown(
        "<div style='text-align:center;padding:12px 0 24px 0;'>"
        "<div style='font-size:28px;'>🌺 🌸 🌼</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:24px;font-weight:800;"
        "color:#1a1a1a;margin-top:8px;'>Why does women's representation matter?</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:14px;color:#555;"
        "max-width:600px;margin:10px auto 0 auto;font-weight:600;line-height:1.7;'>"
        "As of 2026, women make up half the world's population but hold less than a third of its parliamentary seats."
        "</div></div>",
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="fact-card"><div class="fact-icon">🌿</div>
          <div class="fact-number">27.5%</div>
          <div class="fact-label">of parliamentarians worldwide are women — up from just 11% in 1995</div>
          <div class="fact-sub">At this pace, parity won't be reached until 2063</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="fact-card teal"><div class="fact-icon">🌺</div>
          <div class="fact-number teal">7</div>
          <div class="fact-label">countries have achieved 50% or more women in parliament</div>
          <div class="fact-sub">Rwanda leads at 64%, Cuba 57%, Nicaragua 55%</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="fact-card gold"><div class="fact-icon">⏳</div>
          <div class="fact-number gold">130</div>
          <div class="fact-label">years until women hold equal power in top government roles, at the current rate</div>
          <div class="fact-sub">Only 16 countries have a woman Head of State as of Jan 2026</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("""<div class="fact-card blue"><div class="fact-icon">🏛️</div>
          <div class="fact-number blue">22%</div>
          <div class="fact-label">of Cabinet Ministers worldwide are women</div>
          <div class="fact-sub">Most lead social ministries — rarely finance or defence</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown("""<div class="fact-card green"><div class="fact-icon">📋</div>
          <div class="fact-number green">+5pp</div>
          <div class="fact-label">more women elected in countries with legislated gender quotas</div>
          <div class="fact-sub">And +7pp in local government — quotas work</div></div>""", unsafe_allow_html=True)
    with c6:
        st.markdown("""<div class="fact-card"><div class="fact-icon">💧</div>
          <div class="fact-number">62%</div>
          <div class="fact-label">more drinking water projects in villages with women-led councils in India</div>
          <div class="fact-sub">Women in power deliver different — and broader — priorities</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Nunito,sans-serif;font-size:18px;font-weight:800;color:#1a1a1a;margin-bottom:6px;'>🌍 How does your region compare?</div>", unsafe_allow_html=True)
    st.caption("Share of parliamentary seats held by women, by world region (2026)")
    reg_df = pd.DataFrame({
        "region": ["Latin America & Caribbean","Europe & Northern America","Sub-Saharan Africa",
                   "Eastern & South-Eastern Asia","Oceania","Northern Africa & Western Asia","Central & Southern Asia"],
        "pct": [37, 33, 27, 24, 21, 18.5, 17]
    }).sort_values("pct", ascending=True)
    fig_reg = go.Figure(go.Bar(
        x=reg_df["pct"], y=reg_df["region"], orientation="h",
        marker_color=["#0097a7","#0d3320","#fb8c00","#1565c0","#f9c74f","#f4511e","#e8354a"],
        text=reg_df["pct"].apply(lambda x: f"{x}%"), textposition="outside",
        textfont=dict(color="#1a1a1a", size=13, family="Nunito, sans-serif")
    ))
    fig_reg.add_vline(x=50, line_dash="dash", line_color="#e8354a",
                      annotation_text="Gender parity (50%)",
                      annotation_font=dict(color="#e8354a", size=13, family="Nunito, sans-serif"),
                      annotation_position="top")
    lr = base_layout()
    lr.update(dict(xaxis=dict(title="% women parliamentarians", range=[0, 62], **AXIS),
                   yaxis=dict(title="", **AXIS), margin=dict(l=0, r=70, t=20, b=0),
                   height=300, showlegend=False))
    fig_reg.update_layout(**lr)
    st.plotly_chart(fig_reg, use_container_width=True)
    st.divider()
    st.markdown("""<div style="text-align:center;font-family:'Nunito',sans-serif;font-size:13px;color:#888;padding:8px 0;">
    Source: UN Women · <a href="https://www.unwomen.org/en/articles/facts-and-figures/facts-and-figures-womens-leadership-and-political-participation"
    style="color:#e8354a;font-weight:700;">unwomen.org</a> · January 2026</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ══════════════════════════════════════════════════════════════════
with tab_line:
    st.markdown(
        "<div style='font-family:Nunito,sans-serif;font-size:18px;font-weight:800;"
        "color:#1a1a1a;margin-bottom:4px;'>📈 Trends over time</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:13px;color:#666;margin-bottom:16px;'>"
        "Each line shows the % of parliamentary seats held by women after each election. "
        "The dotted grey line is the world average. "
        "Dashed lines with ★ show our model's predictions for the next 3 elections.</div>",
        unsafe_allow_html=True
    )
    if not selected:
        st.info("🌺 Select at least one country from the sidebar to get started.")
    else:
        year_range = st.slider("Year range", 1990, 2045, (1990, 2040), key="line_slider")
        fig = go.Figure()

        wav = world_avg_df[
            (world_avg_df["Year"] >= year_range[0]) &
            (world_avg_df["Year"] <= year_range[1])
        ]
        if not wav.empty:
            fig.add_trace(go.Scatter(
                x=wav["Year"], y=wav["pct"], mode="lines", name="🌍 World average",
                line=dict(color="#bbbbbb", width=2, dash="dot"),
                hovertemplate="World avg %{x}: %{y:.1f}%<extra></extra>"
            ))

        for i, country in enumerate(selected):
            cdf   = df[df["country_name"] == country].sort_values(COL_YEAR)
            if cdf.empty: continue
            color = COLORS[i % len(COLORS)]

            hist = cdf[
                (cdf[COL_IS_FUTURE] == False) &
                (cdf[COL_YEAR] >= year_range[0]) &
                (cdf[COL_YEAR] <= year_range[1])
            ]
            if not hist.empty:
                fig.add_trace(go.Scatter(
                    x=hist[COL_YEAR], y=hist[COL_PCT_WOMEN],
                    mode="lines+markers", name=country,
                    line=dict(color=color, width=2.5), marker=dict(size=6),
                    hovertemplate="%{x}: %{y:.1f}%<extra>" + country + "</extra>"
                ))

            fut = cdf[
                (cdf[COL_IS_FUTURE] == True) &
                (cdf[COL_YEAR] >= year_range[0]) &
                (cdf[COL_YEAR] <= year_range[1])
            ]
            if not fut.empty and not hist.empty:
                bridge_x = [int(hist[COL_YEAR].iloc[-1])] + fut[COL_YEAR].tolist()
                bridge_y = [float(hist[COL_PCT_WOMEN].iloc[-1])] + fut[COL_PCT_WOMEN].tolist()
                fig.add_trace(go.Scatter(
                    x=bridge_x, y=bridge_y,
                    mode="lines+markers", showlegend=False,
                    line=dict(color=color, width=2.5, dash="dash"),
                    marker=dict(size=9, symbol="star"),
                    hovertemplate="%{x}: %{y:.1f}% (predicted)<extra>" + country + "</extra>"
                ))

        layout = base_layout()
        layout.update(dict(
            xaxis=dict(title="Year", **AXIS),
            yaxis=dict(title="% of seats held by women", range=[0, 80], **AXIS),
        ))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown(
            "<div style='font-family:Nunito,sans-serif;font-size:16px;font-weight:800;"
            "color:#1a1a1a;margin-bottom:12px;'>🌺 Predicted values at next election</div>",
            unsafe_allow_html=True
        )
        cols = st.columns(min(len(selected), 5))
        for i, country in enumerate(selected):
            fut_c  = df[(df["country_name"] == country) & (df[COL_IS_FUTURE] == True)].sort_values(COL_YEAR)
            hist_c = df[(df["country_name"] == country) & (df[COL_IS_FUTURE] == False)].sort_values(COL_YEAR)
            if fut_c.empty or hist_c.empty: continue
            curr_pct = float(hist_c.iloc[-1][COL_PCT_WOMEN])
            fut_pct  = float(fut_c.iloc[0][COL_PCT_WOMEN])
            try:    fut_total = int(fut_c.iloc[0][COL_PRED_TOTAL])
            except: fut_total = None
            with cols[i % 5]:
                st.metric(
                    label=f"{country} ({int(fut_c.iloc[0][COL_YEAR])})",
                    value=f"{fut_total} women" if fut_total else f"{fut_pct:.1f}%",
                    delta=f"{fut_pct - curr_pct:+.1f} pp"
                )

# ══════════════════════════════════════════════════════════════════
# TAB 3 — SNAPSHOT BAR
# ══════════════════════════════════════════════════════════════════
with tab_bar:
    st.markdown(
        "<div style='font-family:Nunito,sans-serif;font-size:18px;font-weight:800;"
        "color:#1a1a1a;margin-bottom:4px;'>📊 Country snapshot</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:13px;color:#666;margin-bottom:16px;'>"
        "Each bar shows the % of parliamentary seats held by women in the selected year. "
        "Bars marked ★ show model predictions. Drag the slider to travel through time.</div>",
        unsafe_allow_html=True
    )
    if not selected:
        st.info("🌺 Select at least one country from the sidebar.")
    else:
        bar_year = st.slider("📅 Select a year", 1990, 2045, 2024, key="bar_slider")
        bar_data = []
        for i, country in enumerate(selected):
            row = filled_df[(filled_df["country_name"] == country) & (filled_df["Year"] == bar_year)]
            if row.empty: continue
            r = row.iloc[0]
            if pd.isna(r["pct"]): continue
            bar_data.append({
                "country":       country,
                "pct":           r["pct"],
                "pred_total":    r["pred_total"],
                "last_election": r["last_election"],
                "is_predicted":  r["is_predicted"],
                "color":         COLORS[i % len(COLORS)]
            })
        if not bar_data:
            st.caption("No data available for the selected countries in this year.")
        else:
            bar_df = pd.DataFrame(bar_data).sort_values("pct", ascending=True)
            bar_df["label"] = bar_df.apply(
                lambda r: f"{r['pct']:.1f}% ★" if r["is_predicted"] else f"{r['pct']:.1f}%", axis=1
            )
            bar_df["hover_note"] = bar_df["is_predicted"].apply(
                lambda x: "Predicted ★" if x else "Historical"
            )
            fig_bar = go.Figure(go.Bar(
                x=bar_df["pct"], y=bar_df["country"], orientation="h",
                marker_color=bar_df["color"].tolist(),
                text=bar_df["label"], textposition="outside",
                textfont=dict(color="#1a1a1a", size=13, family="Nunito, sans-serif"),
                customdata=list(zip(bar_df["last_election"], bar_df["hover_note"])),
                hovertemplate="%{y}: %{x:.1f}%<br>Election: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"
            ))
            lb = base_layout()
            lb.update(dict(
                xaxis=dict(title="% of seats held by women", range=[0, 90], **AXIS),
                yaxis=dict(title="", **AXIS),
                margin=dict(l=0, r=90, t=20, b=0),
                height=max(300, len(bar_df) * 55), hovermode="y unified"
            ))
            fig_bar.update_layout(**lb)
            st.plotly_chart(fig_bar, use_container_width=True)
            note = f"Showing most recent election result by {bar_year}."
            if bar_year > 2025: note += " ★ = model prediction."
            st.caption(note)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — MAP
# ══════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown(
        "<div style='font-family:Nunito,sans-serif;font-size:18px;font-weight:800;"
        "color:#1a1a1a;margin-bottom:4px;'>🗺️ World map</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:13px;color:#666;margin-bottom:16px;'>"
        "Darker green = more women in parliament. Each country shows the result of their most "
        "recent election by the selected year. After 2025, predicted values are shown. "
        "Hover to see exact figures.</div>",
        unsafe_allow_html=True
    )
    map_year = st.slider("📅 Select a year", 1990, 2045, 2024, key="map_slider")
    map_snap = filled_df[filled_df["Year"] == map_year].copy()
    map_snap = map_snap[map_snap["pct"].notna()]
    map_snap["hover_note"] = map_snap["is_predicted"].apply(
        lambda x: "Predicted ★" if x else "Historical"
    )
    if map_snap.empty:
        st.caption("No data available for this year.")
    else:
        fig_map = go.Figure(go.Choropleth(
            locations=map_snap["ISO3"], z=map_snap["pct"], text=map_snap["country_name"],
            customdata=list(zip(map_snap["last_election"], map_snap["hover_note"])),
            colorscale=[[0,"#fffbf2"],[0.2,"#f9c74f"],[0.5,"#f4511e"],[0.75,"#e8354a"],[1,"#0d3320"]],
            colorbar=dict(
                title=dict(text="% women", font=dict(color="#1a1a1a", size=13, family="Nunito, sans-serif")),
                tickfont=dict(color="#1a1a1a", size=12, family="Nunito, sans-serif"),
            ),
            hovertemplate="<b>%{text}</b><br>%{z:.1f}% women in parliament<br>Election: %{customdata[0]} · %{customdata[1]}<extra></extra>"
        ))
        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True,
                     projection_type="natural earth", bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(bgcolor="#fffbf2", bordercolor="#cccccc",
                            font=dict(color="#1a1a1a", size=13, family="Nunito, sans-serif")),
            font=dict(family="Nunito, sans-serif", color="#1a1a1a", size=13), height=520
        )
        st.plotly_chart(fig_map, use_container_width=True)
        note = f"Showing most recent election result available by {map_year}."
        if map_year > 2025: note += " ★ Countries showing predicted values."
        st.caption(note)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — FEATURE EXPLORER + SHAP
# ══════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown(
        "<div style='font-family:Nunito,sans-serif;font-size:18px;font-weight:800;"
        "color:#1a1a1a;margin-bottom:4px;'>🔍 What drives the prediction?</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:13px;color:#666;margin-bottom:20px;'>"
        "Select a country to load its current values, then adjust the sliders to explore "
        "how each factor affects how many women will be elected next time.</div>",
        unsafe_allow_html=True
    )
    st.markdown("**Start with a country**")
    explorer_country = st.selectbox("Country", all_countries, key="explorer_country",
                                    label_visibility="collapsed")
    cdf_ex  = df[df["country_name"] == explorer_country].sort_values(COL_YEAR)
    hist_ex = cdf_ex[cdf_ex[COL_IS_FUTURE] == False]
    last_ex = hist_ex.iloc[-1] if not hist_ex.empty else pd.Series()

    def gv(col, default):
        try:
            v = float(last_ex.get(col, default))
            return default if pd.isna(v) else v
        except: return default

    st.markdown("<br>", unsafe_allow_html=True)
    col_sl, col_res = st.columns([1, 1], gap="large")

    with col_sl:
        st.markdown("**Adjust the factors below**")
        vals = {}
        for key, info in FEATURE_INFO.items():
            default_val = gv(info["col"], info["default"])
            if info["step"] == 1 and info["max"] <= 10:
                v = st.slider(f"{info['icon']} {info['label']}",
                              int(info["min"]), int(info["max"]), int(default_val),
                              int(info["step"]), key=f"sl_{key}")
            else:
                v = st.slider(f"{info['icon']} {info['label']}",
                              float(info["min"]), float(info["max"]), float(default_val),
                              float(info["step"]), key=f"sl_{key}")
            st.caption(info["desc"])
            vals[info["col"]] = v
        for lag_col in LAG_FEATURES:
            vals[lag_col] = gv(lag_col, 0.0)

    with col_res:
        pred  = run_prediction(vals)
        shaps = run_shap(vals)

        st.markdown("**Our prediction**")
        if pred is not None:
            color  = "#0d3320" if pred["is_positive"] else "#e8354a"
            abs_pp = abs(pred["pct_change"])
            word   = "more" if pred["is_positive"] else "fewer"
            arrow  = "▲" if pred["is_positive"] else "▼"
            st.markdown(
                f"<div style='background:white;border-radius:18px;padding:28px 24px;"
                f"border-top:5px solid {color};box-shadow:0 4px 20px rgba(0,0,0,0.07);margin-bottom:16px;'>"
                f"<div style='font-family:Nunito,sans-serif;font-size:12px;font-weight:700;"
                f"color:#aaa;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>{explorer_country}</div>"
                f"<div style='font-family:Nunito,sans-serif;font-size:52px;font-weight:800;"
                f"color:{color};line-height:1;'>{arrow} {abs_pp:.1f} pp</div>"
                f"<div style='font-family:Nunito,sans-serif;font-size:16px;font-weight:700;"
                f"color:#1a1a1a;margin-top:10px;'>We predict "
                f"<strong style='color:{color};'>{abs_pp:.1f} percentage points {word}</strong> women elected</div>"
                f"<div style='font-family:Nunito,sans-serif;font-size:13px;color:#888;margin-top:6px;'>"
                f"at the next parliamentary election, compared to today</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background:white;border-radius:18px;padding:28px 24px;"
                "border-top:5px solid #f9c74f;box-shadow:0 4px 20px rgba(0,0,0,0.06);'>"
                "<div style='font-family:Nunito,sans-serif;font-size:14px;font-weight:600;color:#888;'>"
                "Adjust the sliders on the left to see how changes in each factor "
                "would affect the predicted outcome for this country.<br><br>"
                "<span style='font-size:12px;color:#bbb;'>Live predictions require the trained model file.</span>"
                "</div></div>",
                unsafe_allow_html=True
            )

        st.markdown(
            "<div style='font-family:Nunito,sans-serif;font-size:15px;font-weight:800;"
            "color:#1a1a1a;margin:16px 0 4px 0;'>Which factors matter most?</div>"
            "<div style='font-family:Nunito,sans-serif;font-size:12px;color:#888;margin-bottom:10px;'>"
            "Green = pushing toward more women elected. Red = pushing toward fewer. "
            "Length shows how strong the influence is.</div>",
            unsafe_allow_html=True
        )

        if shaps is not None:
            shap_df = pd.DataFrame([
                {"factor": FEATURE_LABELS.get(k, k), "value": float(v)}
                for k, v in shaps.items()
            ]).sort_values("value", ascending=True)
            bar_colors = ["#e8354a" if v < 0 else "#0d3320" for v in shap_df["value"]]
            text_vals  = shap_df["value"].apply(lambda v: f"{v:+.3f}")
        else:
            shap_df = pd.DataFrame({
                "factor": list(FEATURE_LABELS.values()),
                "value":  [0.0] * len(FEATURE_LABELS)
            })
            bar_colors = ["#dddddd"] * len(shap_df)
            text_vals  = ["— load model to see"] * len(shap_df)

        fig_shap = go.Figure(go.Bar(
            x=shap_df["value"], y=shap_df["factor"], orientation="h",
            marker_color=bar_colors,
            text=text_vals, textposition="outside",
            textfont=dict(color="#1a1a1a", size=11, family="Nunito, sans-serif"),
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>"
        ))
        fig_shap.add_vline(x=0, line_color="#cccccc", line_width=1.5)
        ls = base_layout()
        ls.update(dict(
            xaxis=dict(title="Influence on prediction", range=[-1, 1], **AXIS),
            yaxis=dict(title="", **AXIS),
            margin=dict(l=0, r=130, t=10, b=0),
            height=320, showlegend=False, hovermode="y unified"
        ))
        fig_shap.update_layout(**ls)
        st.plotly_chart(fig_shap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 6 — INSIGHTS PLACEHOLDER
# ══════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown(
        "<div style='text-align:center;padding:40px 0 20px 0;'>"
        "<div style='font-size:48px;'>🌏</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:24px;font-weight:800;"
        "color:#1a1a1a;margin-top:12px;'>Model Insights</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:14px;color:#666;"
        "max-width:520px;margin:12px auto 0 auto;font-weight:600;line-height:1.7;'>"
        "Once our final model is trained, this tab will surface the most interesting patterns — "
        "regional trends, the impact of quotas, which countries are set to make the biggest gains, and more."
        "</div></div>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    placeholder_insights = [
        ("🌏", "Asia is predicted to see the largest regional growth in female parliamentary representation by 2030.", "Coming soon — based on model predictions"),
        ("📋", "Countries that introduced strong gender quotas after 2010 show 2× the average growth rate.", "Coming soon — based on model predictions"),
        ("🌐", "Internet access above 80% is strongly associated with faster gains in women's representation.", "Coming soon — based on model predictions"),
        ("⚖️", "The countries with the lowest Gender Inequality Index also have the most consistent representation growth.", "Coming soon — based on model predictions"),
    ]
    c1, c2 = st.columns(2)
    for idx, (icon, text, sub) in enumerate(placeholder_insights):
        col = c1 if idx % 2 == 0 else c2
        with col:
            st.markdown(
                f"<div style='background:white;border-radius:16px;padding:24px;"
                f"border-left:5px solid #f9c74f;box-shadow:0 3px 16px rgba(0,0,0,0.06);margin-bottom:14px;'>"
                f"<div style='font-size:28px;margin-bottom:8px;'>{icon}</div>"
                f"<div style='font-family:Nunito,sans-serif;font-size:14px;font-weight:700;"
                f"color:#1a1a1a;line-height:1.6;'>{text}</div>"
                f"<div style='font-family:Nunito,sans-serif;font-size:12px;color:#aaa;"
                f"margin-top:8px;font-style:italic;'>{sub}</div></div>",
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════
# TAB 7 — GEMINI CHAT
# ══════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(
        "<div style='font-family:Nunito,sans-serif;font-size:18px;font-weight:800;"
        "color:#1a1a1a;margin-bottom:4px;'>💬 Ask the data</div>"
        "<div style='font-family:Nunito,sans-serif;font-size:13px;color:#666;margin-bottom:20px;'>"
        "Ask anything about women's parliamentary representation — which countries are improving fastest, "
        "how quotas affect outcomes, what our model predicts for a specific region, and more. "
        "Answers are based only on the data powering this dashboard.</div>",
        unsafe_allow_html=True
    )

    for msg in st.session_state.chat_history:
        css_class = "chat-user" if msg["role"] == "user" else "chat-ai"
        st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask a question about the data...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            gemini = genai.GenerativeModel(GEMINI_MODEL)
            system_prompt = f"""You are a data analyst for the CinnaRollz dashboard,
which tracks and predicts women's parliamentary representation across 162 countries.

Keep answers concise, factual, and accessible to a non-technical audience.
Use specific numbers from the data whenever possible.
Do not speculate beyond what the data shows.

When asked to aggregate, compare, or rank countries or regions:
- Use the numbers provided to compute averages, totals, or rankings yourself
- You CAN sum or average individual country values to answer regional questions
- Always show your reasoning briefly (e.g. "averaging the 12 Asian countries in the data...")

If asked about a specific year with no data, use the closest available year to the year requested and say so.
If a question truly cannot be answered from this data, say so in one sentence.

DATA:
{chat_ctx}
"""
            history_text = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in st.session_state.chat_history[-6:]
            ])
            full_prompt = f"{system_prompt}\n\nCONVERSATION:\n{history_text}"
            response    = gemini.generate_content(full_prompt)
            answer      = response.text

        except KeyError:
            answer = "⚠️ Gemini API key not found. Add it to `.env` as `GEMINI_API_KEY = 'your-key-here'`"
        except Exception as e:
            answer = f"⚠️ Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    if not st.session_state.chat_history:
        st.markdown(
            "<div style='font-family:Nunito,sans-serif;font-size:13px;color:#aaa;"
            "text-align:center;margin-top:40px;'>"
            "Try asking: <em>\"Which country is predicted to improve the most?\"</em> or "
            "<em>\"How do quotas affect representation in Africa?\"</em></div>",
            unsafe_allow_html=True
        )

# ── footer ────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;font-family:'Nunito',sans-serif;font-size:13px;color:#888;padding:4px 0;">
🌺 Data: IPU Parline &nbsp;·&nbsp; Model: XGBoost &nbsp;·&nbsp; CinnaRollz 2026 &nbsp;·&nbsp; 🌿 🥐 🌸
</div>
""", unsafe_allow_html=True)
