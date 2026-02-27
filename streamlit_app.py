import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Pharmacy KPI Dashboard", layout="wide", page_icon="💊")

STORE_COLORS = {
    'CARLINGFORD': '#E74C3C', 'EASTWOOD': '#3498DB', 'EPPING OXFORD': '#2ECC71',
    'EPPING RAWSON': '#F39C12', 'ERMINGTON': '#9B59B6', 'GRANVILLE': '#1ABC9C',
    'NORTH PARRAMATTA': '#E67E22', 'PARRAMATTA': '#34495E', 'PENNANT HILLS': '#E91E63',
    'SYDNEY OLYMPIC PARK': '#00BCD4', 'WESTFIELD PARRAMATTA': '#8BC34A'
}

GC_TARGET = 0.80
NYX_TARGET = 0.40
NUTRA_TARGET = 0.08

# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_from_gsheet(sheet_url, sheet_name="tblRaw"):
    """Load data from a public Google Sheet"""
    # Convert share URL to CSV export URL
    if '/edit' in sheet_url:
        base = sheet_url.split('/edit')[0]
    elif '/d/' in sheet_url:
        base = sheet_url.split('?')[0]
    else:
        base = sheet_url
    csv_url = base + '/gviz/tq?tqx=out:csv&sheet=' + sheet_name.replace(' ', '%20')
    df = pd.read_csv(csv_url)
    return df

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def prepare_data(df):
    """Clean and prepare the dataframe"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df['Store'] = df['Store'].str.strip()
    df['Metric'] = df['Metric'].str.strip()
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['MonthDate'] = pd.to_datetime(df['MonthDate'], errors='coerce')
    df = df.dropna(subset=['MonthDate', 'Amount'])
    df['YearMonth'] = df['MonthDate'].dt.to_period('M')
    df['Year'] = df['MonthDate'].dt.year
    df['Month'] = df['MonthDate'].dt.month
    df['MonthLabel'] = df['MonthDate'].dt.strftime('%b %Y')
    
    # Normalize Generic Conversion (some stored as whole %, some as decimal)
    mask_gc = (df['Metric'] == 'Generic Conversion') & (df['Amount'] > 1)
    df.loc[mask_gc, 'Amount'] = df.loc[mask_gc, 'Amount'] / 100
    
    # Same for Drug Scans
    mask_ds = (df['Metric'] == 'Drug Scans') & (df['Amount'] > 1)
    df.loc[mask_ds, 'Amount'] = df.loc[mask_ds, 'Amount'] / 100
    
    return df

def get_metric(df, store, month_period, metric):
    """Get a single metric value for a store and month"""
    mask = (df['Store'] == store) & (df['YearMonth'] == month_period) & (df['Metric'] == metric)
    vals = df.loc[mask, 'Amount']
    return vals.iloc[0] if len(vals) > 0 else None

def get_ly_period(period):
    """Get the same month last year"""
    return pd.Period(year=period.year - 1, month=period.month, freq='M')

def fmt_dollar(v):
    if v is None or pd.isna(v): return "—"
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000: return f"${v/1_000:.0f}K"
    return f"${v:.0f}"

def fmt_pct(v):
    if v is None or pd.isna(v): return "—"
    return f"{v*100:.1f}%"

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — DATA SOURCE
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("💊 Pharmacy KPI")
    st.divider()
    
    df = None
    has_secrets = hasattr(st, 'secrets') and 'sheet_url' in st.secrets
    
    if has_secrets:
        # Auto-load from secrets
        data_source = st.radio("Data source", ["Google Sheet (auto)", "Upload CSV"], index=0)
        if data_source == "Google Sheet (auto)":
            try:
                sheet_url = st.secrets["sheet_url"]
                sheet_name = st.secrets.get("sheet_name", "tblRaw")
                df = load_from_gsheet(sheet_url, sheet_name)
                st.success("✓ Connected via secrets")
            except Exception as e:
                st.error(f"Connection failed: {e}")
        else:
            uploaded = st.file_uploader("Upload tblRaw CSV", type=['csv'])
            if uploaded:
                df = load_csv(uploaded)
    else:
        data_source = st.radio("Data source", ["Upload CSV", "Google Sheet URL"], index=0)
        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload tblRaw CSV", type=['csv'])
            if uploaded:
                df = load_csv(uploaded)
        else:
            sheet_url = st.text_input("Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")
            sheet_name = st.text_input("Sheet tab name", value="tblRaw")
            if sheet_url:
                try:
                    df = load_from_gsheet(sheet_url, sheet_name)
                    st.success("✓ Connected")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

if df is None:
    st.title("Pharmacy Group KPI Dashboard")
    st.info("👈 Upload a CSV or connect a Google Sheet URL to get started.\n\nFor Google Sheets: make sure the sheet is **shared as 'Anyone with the link can view'**.")
    st.stop()

# ═══════════════════════════════════════════════════════════════
#  PREPARE DATA
# ═══════════════════════════════════════════════════════════════
df = prepare_data(df)
stores = sorted(df['Store'].unique())
periods = sorted(df['YearMonth'].unique())
period_labels = {p: p.strftime('%b %Y') for p in periods}

with st.sidebar:
    st.divider()
    sel_period = st.selectbox("Current month", periods[::-1], format_func=lambda p: period_labels[p])
    ly_period = get_ly_period(sel_period)
    has_ly = ly_period in periods
    
    st.caption(f"LY comparison: {ly_period.strftime('%b %Y')} {'✓' if has_ly else '✗ no data'}")
    st.caption(f"{len(stores)} stores · {len(periods)} months")

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Summary", "💰 Revenue Truth", "📈 Profitability", 
    "💊 Dispensary", "👥 Wages", "🏥 Store Health", "👤 Customer Value"
])

# ═══════════════════════════════════════════════════════════════
#  TAB 1: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Executive Summary — {period_labels[sel_period]}")
    
    compare_ly = st.toggle("Compare vs Last Year", value=True if has_ly else False, disabled=not has_ly)
    
    # Scorecards
    def group_total(metric, period, agg='sum'):
        mask = (df['Metric'] == metric) & (df['YearMonth'] == period)
        if agg == 'sum': return df.loc[mask, 'Amount'].sum()
        return df.loc[mask, 'Amount'].mean()
    
    metrics_list = [
        ("Sales", "sum"), ("GP$", "sum"), ("Customer #", "sum"), ("Scripts", "sum")
    ]
    
    cols = st.columns(4)
    for i, (metric, agg) in enumerate(metrics_list):
        curr = group_total(metric, sel_period, agg)
        with cols[i]:
            if compare_ly and has_ly:
                ly_val = group_total(metric, ly_period, agg)
                delta = (curr - ly_val) / ly_val if ly_val else None
                delta_str = f"{delta*100:+.1f}% vs LY" if delta is not None else None
                st.metric(metric, fmt_dollar(curr) if '$' in metric or metric == 'Sales' else f"{curr:,.0f}", delta_str)
            else:
                st.metric(metric, fmt_dollar(curr) if '$' in metric or metric == 'Sales' else f"{curr:,.0f}")
    
    st.divider()
    
    col_chart, col_trend = st.columns([1, 1])
    
    with col_chart:
        st.markdown("**Sales by Store**")
        chart_df = df[(df['Metric'] == 'Sales') & (df['YearMonth'] == sel_period)].copy()
        
        if compare_ly and has_ly:
            ly_df = df[(df['Metric'] == 'Sales') & (df['YearMonth'] == ly_period)].copy()
            ly_df = ly_df.rename(columns={'Amount': 'LY'})
            chart_df = chart_df.merge(ly_df[['Store', 'LY']], on='Store', how='left')
            chart_df['Growth'] = (chart_df['Amount'] - chart_df['LY']) / chart_df['LY']
            chart_df = chart_df.sort_values('Amount', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(y=chart_df['Store'], x=chart_df['LY'], name='LY', orientation='h', 
                                 marker_color='#BDC3C7', opacity=0.6))
            fig.add_trace(go.Bar(y=chart_df['Store'], x=chart_df['Amount'], name='Current', orientation='h',
                                 marker_color=[STORE_COLORS.get(s, '#3498DB') for s in chart_df['Store']]))
            fig.update_layout(barmode='overlay', height=400, margin=dict(l=0,r=0,t=10,b=0),
                              legend=dict(orientation='h', y=1.02), yaxis_tickfont_size=10)
        else:
            chart_df = chart_df.sort_values('Amount', ascending=True)
            fig = px.bar(chart_df, y='Store', x='Amount', orientation='h',
                         color='Store', color_discrete_map=STORE_COLORS)
            fig.update_layout(showlegend=False, height=400, margin=dict(l=0,r=0,t=10,b=0),
                              yaxis_tickfont_size=10)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_trend:
        st.markdown("**Group Sales Trend**")
        trend = df[df['Metric'] == 'Sales'].groupby('YearMonth')['Amount'].sum().reset_index()
        trend['Label'] = trend['YearMonth'].apply(lambda p: p.strftime('%b %y'))
        trend = trend.sort_values('YearMonth')
        fig = px.area(trend, x='Label', y='Amount', labels={'Amount': 'Sales', 'Label': ''})
        fig.update_traces(fill='tozeroy', line_color='#3498DB')
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 2: REVENUE TRUTH
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Revenue Truth — {period_labels[sel_period]}")
    st.caption("Are we really growing, or are impactors flattering the numbers?")
    
    impactors = ['Weight Loss Drugs', 'High Price Drugs', 'Bulk Figures', 'CWC']
    
    rev_data = []
    for s in stores:
        sales = get_metric(df, s, sel_period, 'Sales')
        if sales is None or sales == 0: continue
        wld = get_metric(df, s, sel_period, 'Weight Loss Drugs') or 0
        hpd = get_metric(df, s, sel_period, 'High Price Drugs') or 0
        bulk = get_metric(df, s, sel_period, 'Bulk Figures') or 0
        cwc = get_metric(df, s, sel_period, 'CWC') or 0
        total_imp = wld + hpd + bulk + cwc
        organic = sales - total_imp
        dep_pct = total_imp / sales if sales else 0
        
        # Wages as % of sales (for combined view)
        wage = get_metric(df, s, sel_period, 'Wage Fortnight Actual') or 0
        wage_pct = (wage * 2) / sales if sales else 0  # fortnight → monthly approx
        
        rev_data.append({
            'Store': s, 'Sales': sales, 'Organic': organic,
            'Weight Loss Drugs': wld, 'High Price Drugs': hpd,
            'Bulk Figures': bulk, 'CWC': cwc,
            'Impactor Total': total_imp, 'Impactor %': dep_pct,
            'Wage %': wage_pct
        })
    
    rev_df = pd.DataFrame(rev_data).sort_values('Impactor %', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Sales Breakdown — Impactors vs Organic**")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=rev_df['Store'], x=rev_df['Organic'], name='Organic Sales', 
                             orientation='h', marker_color='#27AE60'))
        fig.add_trace(go.Bar(y=rev_df['Store'], x=rev_df['Weight Loss Drugs'], name='Weight Loss', 
                             orientation='h', marker_color='#E74C3C'))
        fig.add_trace(go.Bar(y=rev_df['Store'], x=rev_df['High Price Drugs'], name='High Price', 
                             orientation='h', marker_color='#F39C12'))
        fig.add_trace(go.Bar(y=rev_df['Store'], x=rev_df['Bulk Figures'], name='Bulk', 
                             orientation='h', marker_color='#9B59B6'))
        fig.add_trace(go.Bar(y=rev_df['Store'], x=rev_df['CWC'], name='CWC', 
                             orientation='h', marker_color='#3498DB'))
        fig.update_layout(barmode='stack', height=450, margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(orientation='h', y=1.05), yaxis_tickfont_size=10)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Impactor Dependency**")
        for _, row in rev_df.iterrows():
            pct = row['Impactor %']
            color = '#E74C3C' if pct > 0.4 else '#F39C12' if pct > 0.25 else '#27AE60'
            st.markdown(f"**{row['Store'][:15]}**: <span style='color:{color};font-size:20px;font-weight:700'>{pct*100:.0f}%</span>", 
                        unsafe_allow_html=True)
    
    # Impactor trend over time
    st.divider()
    st.markdown("**Impactor Dependency % Trend by Store**")
    trend_data = []
    for p in periods:
        for s in stores:
            sales = get_metric(df, s, p, 'Sales')
            if not sales: continue
            imp_total = sum(get_metric(df, s, p, m) or 0 for m in impactors)
            trend_data.append({'Period': p.strftime('%b %y'), 'Store': s, 'Impactor %': imp_total / sales, 'Sort': p})
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data).sort_values('Sort')
        fig = px.line(trend_df, x='Period', y='Impactor %', color='Store', color_discrete_map=STORE_COLORS)
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0), yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 3: PROFITABILITY
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Profitability — {period_labels[sel_period]} vs {ly_period.strftime('%b %Y')}")
    
    if not has_ly:
        st.warning(f"No LY data available for {ly_period.strftime('%b %Y')}. Need same month last year in tblRaw.")
    else:
        st.caption("Top-right = growing sales AND margin. Bottom-right = growing sales but shrinking margin (red flag).")
        
        scatter_data = []
        for s in stores:
            s_curr = get_metric(df, s, sel_period, 'Sales')
            s_ly = get_metric(df, s, ly_period, 'Sales')
            gp_curr = get_metric(df, s, sel_period, 'GP$')
            gp_ly = get_metric(df, s, ly_period, 'GP$')
            
            if not all([s_curr, s_ly, gp_curr, gp_ly]): continue
            
            scatter_data.append({
                'Store': s,
                'Sales Growth YoY': (s_curr - s_ly) / s_ly,
                'GP$ Growth YoY': (gp_curr - gp_ly) / gp_ly,
                'Sales': s_curr
            })
        
        if scatter_data:
            sc_df = pd.DataFrame(scatter_data)
            fig = px.scatter(sc_df, x='Sales Growth YoY', y='GP$ Growth YoY', color='Store',
                             color_discrete_map=STORE_COLORS, size='Sales', size_max=30,
                             hover_data={'Sales': ':$,.0f', 'Sales Growth YoY': ':.1%', 'GP$ Growth YoY': ':.1%'})
            fig.add_hline(y=0, line_dash="dash", line_color="#bdc3c7")
            fig.add_vline(x=0, line_dash="dash", line_color="#bdc3c7")
            
            # Quadrant labels
            fig.add_annotation(x=0.15, y=0.15, text="⭐ Stars", showarrow=False, font=dict(size=12, color='#27AE60'), opacity=0.5)
            fig.add_annotation(x=-0.15, y=-0.15, text="⚠️ Trouble", showarrow=False, font=dict(size=12, color='#E74C3C'), opacity=0.5)
            fig.add_annotation(x=0.15, y=-0.15, text="🚩 Buying Revenue", showarrow=False, font=dict(size=12, color='#F39C12'), opacity=0.5)
            fig.add_annotation(x=-0.15, y=0.15, text="💡 Efficiency", showarrow=False, font=dict(size=12, color='#3498DB'), opacity=0.5)
            
            fig.update_layout(height=500, xaxis_tickformat='.0%', yaxis_tickformat='.0%',
                              margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Stores without LY data
            missing = [s for s in stores if s not in sc_df['Store'].values]
            if missing:
                st.info(f"Stores not plotted (no LY data): {', '.join(missing)}")
        else:
            st.warning("No stores have both current and LY data for comparison.")
    
    st.divider()
    
    # GP% trend
    st.markdown("**GP% Trend — All Stores**")
    gp_trend = df[df['Metric'] == 'GP%'].copy()
    gp_trend['Label'] = gp_trend['YearMonth'].apply(lambda p: p.strftime('%b %y'))
    gp_trend = gp_trend.sort_values('YearMonth')
    fig = px.line(gp_trend, x='Label', y='Amount', color='Store', color_discrete_map=STORE_COLORS)
    fig.add_hline(y=0.15, line_dash="dot", line_color="#E74C3C", annotation_text="15% floor")
    fig.update_layout(height=350, yaxis_tickformat='.0%', yaxis_title='GP%',
                      margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 4: DISPENSARY
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Dispensary — {period_labels[sel_period]}")
    
    # ── Generic Conversion League ──
    st.markdown(f"**Generic Conversion League — Target: {GC_TARGET*100:.0f}%**")
    
    disp_data = []
    for s in stores:
        gc = get_metric(df, s, sel_period, 'Generic Conversion')
        glp = get_metric(df, s, sel_period, 'Generic Loss Profit')
        scripts = get_metric(df, s, sel_period, 'Scripts')
        ds = get_metric(df, s, sel_period, 'Drug Scans')
        nyx = get_metric(df, s, sel_period, 'Nyxoid Units')
        opi = get_metric(df, s, sel_period, 'Opioid Units')
        nutra = get_metric(df, s, sel_period, 'Nutralife Probiotic')
        anti = get_metric(df, s, sel_period, 'Antibiotic Units')
        
        if gc is None: continue
        
        # Calculate potential gain from hitting target
        gap_to_target = max(0, GC_TARGET - gc)
        non_generic_pct = 1 - gc
        potential_gain = (gap_to_target / non_generic_pct * abs(glp)) if (glp and non_generic_pct > 0 and gap_to_target > 0) else 0
        
        nyx_conv = nyx / opi if (nyx is not None and opi and opi > 0) else None
        nutra_conv = nutra / anti if (nutra is not None and anti and anti > 0) else None
        
        disp_data.append({
            'Store': s, 'Generic Conversion': gc, 'Generic Loss Profit': glp or 0,
            'Gap to Target': gap_to_target, 'Potential Gain': potential_gain,
            'Drug Scans': ds, 'Scripts': scripts,
            'Nyxoid Conv': nyx_conv, 'Nutralife Conv': nutra_conv,
            'Nyxoid Units': nyx or 0, 'Opioid Units': opi or 0,
            'Nutralife Units': nutra or 0, 'Antibiotic Units': anti or 0
        })
    
    disp_df = pd.DataFrame(disp_data).sort_values('Generic Conversion', ascending=True)
    
    if len(disp_df) > 0:
        # Bar chart with gain on hover
        colors = ['#27AE60' if gc >= GC_TARGET else '#F39C12' if gc >= 0.65 else '#E74C3C' 
                  for gc in disp_df['Generic Conversion']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=disp_df['Store'], x=disp_df['Generic Conversion'], orientation='h',
            marker_color=colors,
            customdata=np.stack([disp_df['Potential Gain'], disp_df['Gap to Target'], disp_df['Generic Loss Profit']], axis=-1),
            hovertemplate='<b>%{y}</b><br>' +
                          'Generic Conv: %{x:.1%}<br>' +
                          'Gap to 80%: %{customdata[1]:.1%}<br>' +
                          'Generic Loss Profit: $%{customdata[2]:,.0f}<br>' +
                          '<b>Potential gain at 80%: $%{customdata[0]:,.0f}</b><extra></extra>'
        ))
        fig.add_vline(x=GC_TARGET, line_dash="dash", line_color="#27AE60", line_width=2,
                      annotation_text=f"{GC_TARGET*100:.0f}% Target")
        fig.update_layout(height=400, xaxis_tickformat='.0%', margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_range=[0.5, 0.85], yaxis_tickfont_size=10)
        st.plotly_chart(fig, use_container_width=True)
        
        # Opportunity summary
        total_gain = disp_df['Potential Gain'].sum()
        below_target = len(disp_df[disp_df['Generic Conversion'] < GC_TARGET])
        if total_gain > 0:
            st.success(f"💰 **Total opportunity if all stores hit {GC_TARGET*100:.0f}%: {fmt_dollar(total_gain)}** ({below_target} stores below target)")
    
    st.divider()
    
    # ── GC Trend ──
    st.markdown("**Generic Conversion Trend**")
    gc_trend = df[df['Metric'] == 'Generic Conversion'].copy()
    gc_trend['Label'] = gc_trend['YearMonth'].apply(lambda p: p.strftime('%b %y'))
    gc_trend = gc_trend.sort_values('YearMonth')
    fig = px.line(gc_trend, x='Label', y='Amount', color='Store', color_discrete_map=STORE_COLORS)
    fig.add_hline(y=GC_TARGET, line_dash="dash", line_color="#27AE60", annotation_text=f"{GC_TARGET*100:.0f}%")
    fig.update_layout(height=300, yaxis_tickformat='.0%', margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ── Nyxoid & Nutralife Conversion ──
    col_nyx, col_nut = st.columns(2)
    
    with col_nyx:
        st.markdown("**Nyxoid Conversion % (Nyxoid / Opioid Units)**")
        nyx_data = disp_df[disp_df['Nyxoid Conv'].notna()].sort_values('Nyxoid Conv', ascending=True)
        if len(nyx_data) > 0:
            nyx_colors = ['#27AE60' if v >= NYX_TARGET else '#F39C12' if v >= 0.25 else '#E74C3C' 
                          for v in nyx_data['Nyxoid Conv']]
            fig = go.Figure(go.Bar(
                y=nyx_data['Store'], x=nyx_data['Nyxoid Conv'], orientation='h', marker_color=nyx_colors,
                customdata=np.stack([nyx_data['Nyxoid Units'], nyx_data['Opioid Units']], axis=-1),
                hovertemplate='<b>%{y}</b><br>Conv: %{x:.1%}<br>Nyxoid: %{customdata[0]:.0f} / Opioid: %{customdata[1]:.0f}<extra></extra>'
            ))
            fig.add_vline(x=NYX_TARGET, line_dash="dash", line_color="#27AE60", annotation_text=f"{NYX_TARGET*100:.0f}%")
            fig.update_layout(height=350, xaxis_tickformat='.0%', margin=dict(l=0,r=0,t=10,b=0), yaxis_tickfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Nyxoid/Opioid data for this month")
    
    with col_nut:
        st.markdown("**Nutralife Conversion % (Nutralife / Antibiotic Units)**")
        nut_data = disp_df[disp_df['Nutralife Conv'].notna()].sort_values('Nutralife Conv', ascending=True)
        if len(nut_data) > 0:
            nut_colors = ['#27AE60' if v >= NUTRA_TARGET else '#F39C12' if v >= 0.04 else '#E74C3C' 
                          for v in nut_data['Nutralife Conv']]
            fig = go.Figure(go.Bar(
                y=nut_data['Store'], x=nut_data['Nutralife Conv'], orientation='h', marker_color=nut_colors,
                customdata=np.stack([nut_data['Nutralife Units'], nut_data['Antibiotic Units']], axis=-1),
                hovertemplate='<b>%{y}</b><br>Conv: %{x:.1%}<br>Nutralife: %{customdata[0]:.0f} / Antibiotics: %{customdata[1]:.0f}<extra></extra>'
            ))
            fig.add_vline(x=NUTRA_TARGET, line_dash="dash", line_color="#27AE60", annotation_text=f"{NUTRA_TARGET*100:.0f}%")
            fig.update_layout(height=350, xaxis_tickformat='.0%', margin=dict(l=0,r=0,t=10,b=0), yaxis_tickfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Nutralife/Antibiotic data for this month")


# ═══════════════════════════════════════════════════════════════
#  TAB 5: WAGES
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"Wages — {period_labels[sel_period]}")
    
    wage_data = []
    for s in stores:
        wage = get_metric(df, s, sel_period, 'Wage Fortnight Actual')
        wage_g = get_metric(df, s, sel_period, 'Wage Growth %')
        sales = get_metric(df, s, sel_period, 'Sales')
        
        if wage is None: continue
        
        # Wage fortnight → monthly estimate (×2 roughly)
        wage_monthly = wage * 2
        wage_pct_sales = wage_monthly / sales if sales else None
        
        wage_data.append({
            'Store': s, 'Wage Fortnight': wage, 'Wage Monthly Est': wage_monthly,
            'Wage Growth %': wage_g, 'Sales': sales or 0,
            'Wage % of Sales': wage_pct_sales
        })
    
    if wage_data:
        wage_df = pd.DataFrame(wage_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Wages as % of Sales**")
            st.caption("Wage fortnight × 2 ÷ Monthly Sales. Lower is better.")
            w_sorted = wage_df.dropna(subset=['Wage % of Sales']).sort_values('Wage % of Sales', ascending=True)
            if len(w_sorted) > 0:
                colors = ['#27AE60' if v <= 0.08 else '#F39C12' if v <= 0.12 else '#E74C3C' 
                          for v in w_sorted['Wage % of Sales']]
                fig = go.Figure(go.Bar(
                    y=w_sorted['Store'], x=w_sorted['Wage % of Sales'], orientation='h', marker_color=colors,
                    hovertemplate='<b>%{y}</b><br>Wage % of Sales: %{x:.1%}<br><extra></extra>'
                ))
                fig.update_layout(height=400, xaxis_tickformat='.1%', margin=dict(l=0,r=0,t=10,b=0), yaxis_tickfont_size=10)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Wage Growth % by Store**")
            wg = wage_df.dropna(subset=['Wage Growth %']).sort_values('Wage Growth %', ascending=True)
            if len(wg) > 0:
                colors = ['#E74C3C' if v > 0.05 else '#F39C12' if v > 0 else '#27AE60' for v in wg['Wage Growth %']]
                fig = go.Figure(go.Bar(
                    y=wg['Store'], x=wg['Wage Growth %'], orientation='h', marker_color=colors,
                    hovertemplate='<b>%{y}</b><br>Wage Growth: %{x:.1%}<extra></extra>'
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="#bdc3c7")
                fig.update_layout(height=400, xaxis_tickformat='.1%', margin=dict(l=0,r=0,t=10,b=0), yaxis_tickfont_size=10)
                st.plotly_chart(fig, use_container_width=True)
        
        # Wage trend
        st.divider()
        st.markdown("**Wage Fortnight Trend**")
        wage_trend = df[df['Metric'] == 'Wage Fortnight Actual'].copy()
        if len(wage_trend) > 0:
            wage_trend['Label'] = wage_trend['YearMonth'].apply(lambda p: p.strftime('%b %y'))
            wage_trend = wage_trend.sort_values('YearMonth')
            fig = px.line(wage_trend, x='Label', y='Amount', color='Store', color_discrete_map=STORE_COLORS)
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No wage data for this month")


# ═══════════════════════════════════════════════════════════════
#  TAB 6: STORE HEALTH
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.subheader(f"Store Health Matrix — {period_labels[sel_period]} vs {ly_period.strftime('%b %Y')}")
    
    health_data = []
    for s in stores:
        sales_curr = get_metric(df, s, sel_period, 'Sales')
        sales_ly = get_metric(df, s, ly_period, 'Sales') if has_ly else None
        gp_pct = get_metric(df, s, sel_period, 'GP%')
        gc = get_metric(df, s, sel_period, 'Generic Conversion')
        ds = get_metric(df, s, sel_period, 'Drug Scans')
        scripts_curr = get_metric(df, s, sel_period, 'Scripts')
        scripts_ly = get_metric(df, s, ly_period, 'Scripts') if has_ly else None
        cust_curr = get_metric(df, s, sel_period, 'Customer #')
        cust_ly = get_metric(df, s, ly_period, 'Customer #') if has_ly else None
        
        # Impactor dependency
        imp_total = sum(get_metric(df, s, sel_period, m) or 0 for m in ['Weight Loss Drugs','High Price Drugs','Bulk Figures','CWC'])
        imp_pct = imp_total / sales_curr if sales_curr else None
        
        if sales_curr is None: continue
        
        health_data.append({
            'Store': s,
            'Sales YoY': (sales_curr - sales_ly) / sales_ly if sales_ly else None,
            'GP%': gp_pct,
            'Generic Conv': gc,
            'Drug Scans': ds,
            'Scripts YoY': (scripts_curr - scripts_ly) / scripts_ly if scripts_ly else None,
            'Customers YoY': (cust_curr - cust_ly) / cust_ly if cust_ly else None,
            'Impactor %': imp_pct
        })
    
    if health_data:
        health_df = pd.DataFrame(health_data)
        
        # Thresholds: (red_below, green_above)
        thresholds = {
            'Sales YoY': (-0.05, 0.02),
            'GP%': (0.14, 0.17),
            'Generic Conv': (0.65, GC_TARGET),
            'Drug Scans': (0.96, 0.98),
            'Scripts YoY': (-0.05, 0.02),
            'Customers YoY': (-0.05, 0.02),
            'Impactor %': (0.25, 0.40),  # reversed: lower is better
        }
        
        def cell_color(val, metric):
            if val is None or pd.isna(val): return 'background-color: #f0f0f0'
            bad, good = thresholds.get(metric, (0, 0))
            if metric == 'Impactor %':  # reversed
                if val <= bad: return 'background-color: #27AE60; color: white'
                if val <= good: return 'background-color: #F39C12; color: white'
                return 'background-color: #E74C3C; color: white'
            else:
                if val >= good: return 'background-color: #27AE60; color: white'
                if val >= bad: return 'background-color: #F39C12; color: white'
                return 'background-color: #E74C3C; color: white'
        
        # Display as styled table
        display_df = health_df.set_index('Store')
        
        # Format
        format_dict = {
            'Sales YoY': '{:+.1%}', 'GP%': '{:.1%}', 'Generic Conv': '{:.1%}',
            'Drug Scans': '{:.1%}', 'Scripts YoY': '{:+.1%}', 'Customers YoY': '{:+.1%}',
            'Impactor %': '{:.0%}'
        }
        
        styled = display_df.style
        for col in display_df.columns:
            styled = styled.map(lambda v, c=col: cell_color(v, c), subset=[col])
            if col in format_dict:
                styled = styled.format({col: format_dict[col]}, na_rep='—')
        
        st.dataframe(styled, use_container_width=True, height=450)
        
        st.caption("🟢 = Strong | 🟡 = Watch | 🔴 = Needs attention. Impactor % is reversed (lower = better).")
    else:
        st.warning("No data available for health matrix")


# ═══════════════════════════════════════════════════════════════
#  TAB 7: CUSTOMER VALUE
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.subheader(f"Customer Value — {period_labels[sel_period]}")
    st.caption("How much is each customer worth? Higher value per customer = healthier store.")
    
    cv_data = []
    for s in stores:
        sales = get_metric(df, s, sel_period, 'Sales')
        gp = get_metric(df, s, sel_period, 'GP$')
        cust = get_metric(df, s, sel_period, 'Customer #')
        units = get_metric(df, s, sel_period, 'Units Sold #')
        scripts = get_metric(df, s, sel_period, 'Scripts')
        
        if not all([sales, gp, cust]): continue
        
        cv_data.append({
            'Store': s,
            'Sales per Customer': sales / cust if cust else 0,
            'GP$ per Customer': gp / cust if cust else 0,
            'Units per Customer': units / cust if (units and cust) else 0,
            'Scripts per Customer': scripts / cust if (scripts and cust) else 0,
            'Customers': cust, 'Sales': sales, 'GP$': gp
        })
    
    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sales per Customer**")
            sorted_df = cv_df.sort_values('Sales per Customer', ascending=True)
            fig = go.Figure(go.Bar(
                y=sorted_df['Store'], x=sorted_df['Sales per Customer'], orientation='h',
                marker_color=[STORE_COLORS.get(s, '#3498DB') for s in sorted_df['Store']],
                hovertemplate='<b>%{y}</b><br>$%{x:.2f} per customer<extra></extra>'
            ))
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), yaxis_tickfont_size=10,
                              xaxis_tickprefix='$')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**GP$ per Customer**")
            sorted_df = cv_df.sort_values('GP$ per Customer', ascending=True)
            fig = go.Figure(go.Bar(
                y=sorted_df['Store'], x=sorted_df['GP$ per Customer'], orientation='h',
                marker_color=[STORE_COLORS.get(s, '#3498DB') for s in sorted_df['Store']],
                hovertemplate='<b>%{y}</b><br>$%{x:.2f} GP per customer<extra></extra>'
            ))
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), yaxis_tickfont_size=10,
                              xaxis_tickprefix='$')
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Units per Customer** (cross-sell rate)")
            sorted_df = cv_df.sort_values('Units per Customer', ascending=True)
            fig = go.Figure(go.Bar(
                y=sorted_df['Store'], x=sorted_df['Units per Customer'], orientation='h',
                marker_color=[STORE_COLORS.get(s, '#3498DB') for s in sorted_df['Store']],
                hovertemplate='<b>%{y}</b><br>%{x:.1f} units per customer<extra></extra>'
            ))
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), yaxis_tickfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("**Scripts per Customer** (dispensary engagement)")
            sorted_df = cv_df.sort_values('Scripts per Customer', ascending=True)
            fig = go.Figure(go.Bar(
                y=sorted_df['Store'], x=sorted_df['Scripts per Customer'], orientation='h',
                marker_color=[STORE_COLORS.get(s, '#3498DB') for s in sorted_df['Store']],
                hovertemplate='<b>%{y}</b><br>%{x:.2f} scripts per customer<extra></extra>'
            ))
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), yaxis_tickfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter: Customer volume vs customer value
        st.divider()
        st.markdown("**Customer Volume vs Customer Value**")
        st.caption("Top-right = lots of high-value customers (ideal). Bottom-right = lots of low-value customers (volume play). Top-left = few but valuable customers.")
        fig = px.scatter(cv_df, x='Customers', y='GP$ per Customer', color='Store', size='Sales',
                         color_discrete_map=STORE_COLORS, size_max=35,
                         hover_data={'Sales': ':$,.0f', 'GP$ per Customer': ':$.2f', 'Customers': ':,.0f'})
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), xaxis_tickformat=',',
                          yaxis_tickprefix='$')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for customer value analysis")


# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.divider()
st.caption(f"Data: {len(df)} rows · {len(stores)} stores · {periods[0].strftime('%b %Y')} to {periods[-1].strftime('%b %Y')} · Last loaded: {datetime.now().strftime('%H:%M %d/%m/%Y')}")



# ═══════════════════════════════════════════════════════════════
#  FLOATING CHAT WIDGET
#  Uses components.html with a full-viewport transparent iframe.
#  pointer-events:none on the iframe body means clicks pass through
#  to the dashboard below — only the bubble + panel are clickable.
# ═══════════════════════════════════════════════════════════════
import streamlit.components.v1 as components

_ctx = period_labels[sel_period]

_widget = f"""<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
  * {{ box-sizing:border-box; margin:0; padding:0; }}

  /* Make the iframe itself transparent and full-viewport */
  html, body {{
    background: transparent !important;
    overflow: hidden;
    width: 100vw;
    height: 100vh;
    /* Pass ALL clicks through to Streamlit beneath */
    pointer-events: none;
  }}

  /* Re-enable clicks only on our interactive elements */
  #pcbubble, #pcpanel {{ pointer-events: all; }}

  #pcbubble {{
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 54px;
    height: 54px;
    background: #0ea5e9;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 24px rgba(14,165,233,.55);
    border: none;
    transition: all .2s;
    z-index: 9999;
  }}
  #pcbubble:hover {{ background: #38bdf8; transform: scale(1.08); }}
  #pcbubble svg {{ width:24px; height:24px; fill:white; }}
  #pcbadge {{
    position:absolute; top:-2px; right:-2px;
    width:13px; height:13px;
    background:#22c55e; border-radius:50%;
    border:2px solid #060810;
    animation: bpulse 2s infinite;
  }}
  @keyframes bpulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.6;transform:scale(1.2)}} }}

  #pcpanel {{
    position: fixed;
    bottom: 94px;
    right: 28px;
    width: 390px;
    height: 530px;
    background: #0d1117;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    display: none;
    flex-direction: column;
    box-shadow: 0 20px 60px rgba(0,0,0,.8);
    z-index: 9998;
    overflow: hidden;
  }}
  #pcpanel.open {{ display:flex; animation: pup .2s ease; }}
  @keyframes pup {{ from{{opacity:0;transform:translateY(10px)}} to{{opacity:1;transform:translateY(0)}} }}

  #pch {{
    background:#0f1923; padding:12px 16px;
    display:flex; align-items:center; gap:10px;
    border-bottom:1px solid #1e3a5f; flex-shrink:0;
  }}
  .pcdot {{ width:8px;height:8px;background:#22c55e;border-radius:50%;box-shadow:0 0 6px #22c55e;animation:bpulse 2s infinite;flex-shrink:0; }}
  #pch h3 {{ font-family:'IBM Plex Mono',monospace;font-size:12px;color:#e2e8f0;font-weight:500;flex:1; }}
  .pcpill {{ font-family:'IBM Plex Mono',monospace;font-size:9px;background:rgba(14,165,233,.12);color:#38bdf8;padding:2px 8px;border-radius:10px;border:1px solid rgba(14,165,233,.2); }}
  #pcx {{ background:none;border:none;color:#475569;font-size:20px;cursor:pointer;padding:0 4px;line-height:1; }}
  #pcx:hover {{ color:#94a3b8; }}

  #pcmsgs {{ flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px; }}
  #pcmsgs::-webkit-scrollbar {{ width:3px; }}
  #pcmsgs::-webkit-scrollbar-thumb {{ background:#1e3a5f;border-radius:2px; }}

  .pcrow {{ display:flex;flex-direction:column;max-width:90%;animation:pup .15s ease; }}
  .pcrow.user {{ align-self:flex-end;align-items:flex-end; }}
  .pcrow.ai {{ align-self:flex-start;align-items:flex-start; }}
  .pclbl {{ font-family:'IBM Plex Mono',monospace;font-size:9px;color:#334155;margin-bottom:3px;letter-spacing:.08em;text-transform:uppercase; }}
  .pcbbl {{ padding:9px 13px;border-radius:8px;font-size:12.5px;line-height:1.55; }}
  .pcrow.user .pcbbl {{ background:#0c2d48;border:1px solid #1e4a6e;color:#bae6fd; }}
  .pcrow.ai .pcbbl {{ background:#111827;border:1px solid #1e293b;color:#e2e8f0; }}
  .pcbbl strong {{ color:#38bdf8; }}
  .pcbbl code {{ font-family:'IBM Plex Mono',monospace;font-size:11px;background:#1e293b;padding:1px 5px;border-radius:3px;color:#7dd3fc; }}
  .pcbbl table {{ border-collapse:collapse;width:100%;margin:6px 0;font-size:11px;font-family:'IBM Plex Mono',monospace; }}
  .pcbbl th {{ background:#1e293b;color:#38bdf8;padding:4px 8px;border:1px solid #334155;text-align:left;font-size:10px; }}
  .pcbbl td {{ padding:3px 8px;border:1px solid #1e293b;color:#cbd5e1; }}
  .pcbbl tr:nth-child(even) td {{ background:#0d1117; }}
  .pcbbl ul,.pcbbl ol {{ padding-left:16px;margin:4px 0; }}
  .pcbbl li {{ margin:2px 0; }}
  .pcthink {{ display:flex;gap:4px;padding:10px 13px; }}
  .pcthink span {{ width:5px;height:5px;background:#38bdf8;border-radius:50%;opacity:.3;animation:ptk 1.2s infinite; }}
  .pcthink span:nth-child(2) {{ animation-delay:.2s; }}
  .pcthink span:nth-child(3) {{ animation-delay:.4s; }}
  @keyframes ptk {{ 0%,100%{{opacity:.3;transform:scale(1)}} 50%{{opacity:1;transform:scale(1.4)}} }}

  #pcsugg {{ padding:0 14px 10px;display:flex;flex-wrap:wrap;gap:5px;flex-shrink:0; }}
  .pcsbtn {{ background:rgba(14,165,233,.07);border:1px solid rgba(14,165,233,.2);color:#7dd3fc;font-size:10px;padding:4px 10px;border-radius:12px;cursor:pointer;font-family:'IBM Plex Sans',sans-serif; }}
  .pcsbtn:hover {{ background:rgba(14,165,233,.15); }}

  #pcinrow {{ padding:10px 12px;border-top:1px solid #1e293b;display:flex;gap:7px;align-items:center;background:#0f1923;flex-shrink:0; }}
  #pcinput {{ flex:1;background:#0d1117;border:1px solid #1e3a5f;border-radius:6px;color:#e2e8f0;font-size:12px;padding:8px 12px;outline:none;font-family:'IBM Plex Sans',sans-serif; }}
  #pcinput:focus {{ border-color:#0ea5e9; }}
  #pcinput::placeholder {{ color:#334155; }}
  #pcsend {{ background:#0ea5e9;border:none;color:#000;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:500;padding:8px 12px;border-radius:6px;cursor:pointer; }}
  #pcsend:hover {{ background:#38bdf8; }}
  #pcsend:disabled {{ background:#1e293b;color:#334155;cursor:not-allowed; }}
</style>
</head>
<body>

<button id="pcbubble" onclick="pctoggle()">
  <svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-3 10H7v-2h10v2zm0-3H7V7h10v2z"/></svg>
  <div id="pcbadge"></div>
</button>

<div id="pcpanel">
  <div id="pch">
    <div class="pcdot"></div>
    <h3>Pharmacy AI</h3>
    <span class="pcpill">{_ctx}</span>
    <button id="pcx" onclick="pctoggle()">×</button>
  </div>
  <div id="pcmsgs">
    <div class="pcrow ai">
      <div class="pclbl">pharmacy ai</div>
      <div class="pcbbl">Hey — I can see you're on <strong>{_ctx}</strong>. Ask me anything about this period, any store, or the whole network.</div>
    </div>
  </div>
  <div id="pcsugg">
    <button class="pcsbtn" onclick="pcq('Which stores are below GP% target this month?')">GP% laggards</button>
    <button class="pcsbtn" onclick="pcq('Dispensary safety index all stores')">Disp safety</button>
    <button class="pcsbtn" onclick="pcq('Best sales growth vs LY?')">Best growth</button>
    <button class="pcsbtn" onclick="pcq('Stores over wage budget?')">Wage pressure</button>
  </div>
  <div id="pcinrow">
    <input id="pcinput" type="text" placeholder="Ask about {_ctx}…">
    <button id="pcsend">SEND</button>
  </div>
</div>

<script>
const s=document.createElement('script');
s.src='https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js';
document.head.appendChild(s);

const API='http://127.0.0.1:5000/ask';
const CTX='{_ctx}';
let hist=[],open=false,suggShown=true;

function pctoggle(){{
  open=!open;
  document.getElementById('pcpanel').classList.toggle('open',open);
  if(open){{ document.getElementById('pcbadge').style.display='none'; document.getElementById('pcinput').focus(); }}
}}

function pcq(q){{
  document.getElementById('pcinput').value=q;
  pcHideS();
  pcsend();
}}

function pcHideS(){{
  if(suggShown){{ document.getElementById('pcsugg').style.display='none'; suggShown=false; }}
}}

document.getElementById('pcinput').addEventListener('keydown',e=>{{ if(e.key==='Enter') pcsend(); }});
document.getElementById('pcsend').addEventListener('click', pcsend);

async function pcsend(){{
  const inp=document.getElementById('pcinput');
  const q=inp.value.trim(); if(!q) return;
  pcHideS();
  pcadd('user',q); inp.value='';
  document.getElementById('pcsend').disabled=true;
  const t=pcthink();
  try{{
    const cq=hist.length===0?'[Dashboard context: viewing '+CTX+'] '+q:q;
    const r=await fetch(API,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{question:cq,history:hist}})}});
    const d=await r.json();
    t.remove(); pcadd('ai',d.answer||'No response.'); hist=d.history||hist;
  }}catch(e){{
    t.remove(); pcadd('ai','⚠️ Cannot reach the API. Make sure Flask is running on localhost:5000.');
  }}
  document.getElementById('pcsend').disabled=false; inp.focus();
}}

function pcadd(role,text){{
  const m=document.getElementById('pcmsgs');
  const r=document.createElement('div'); r.className='pcrow '+role;
  const lbl=role==='user'?'you':'pharmacy ai';
  const html=(typeof marked!=='undefined')?marked.parse(text):text.replace(/\\n/g,'<br>');
  r.innerHTML='<div class="pclbl">'+lbl+'</div><div class="pcbbl">'+html+'</div>';
  m.appendChild(r); m.scrollTop=m.scrollHeight; return r;
}}

function pcthink(){{
  const m=document.getElementById('pcmsgs');
  const r=document.createElement('div'); r.className='pcrow ai';
  r.innerHTML='<div class="pclbl">pharmacy ai</div><div class="pcbbl"><div class="pcthink"><span></span><span></span><span></span></div></div>';
  m.appendChild(r); m.scrollTop=m.scrollHeight; return r;
}}
</script>
</body></html>"""

components.html(_widget, height=600, scrolling=False)
