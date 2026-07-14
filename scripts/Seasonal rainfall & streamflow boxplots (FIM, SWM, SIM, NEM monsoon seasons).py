import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from scipy.stats import linregress
from matplotlib.ticker import StrMethodFormatter
from matplotlib.patches import Patch

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "legend.edgecolor": "black",
    "boxplot.flierprops.color": "black",
    "boxplot.whiskerprops.color": "black",
    "boxplot.boxprops.color": "black",
    "boxplot.capprops.color": "black",
    "boxplot.medianprops.color": "black"
})

output_dir = r"D:\Research Datasets\Additional Datasets\Working Folder\Rainfall Variations\Plots"
os.makedirs(output_dir, exist_ok=True)

rf_stations = [
    {"name": "Deraniyagala",  "file": r"D:\...\Deraniyagala_RF_1990-2015.xlsx"},
    {"name": "Nawalapitiya",  "file": r"D:\...\Nawalapitiya_RF_1990-2015.xlsx"},
    {"name": "Wellawaya",     "file": r"D:\...\Welawaya_RF_1990_2015.xlsx"},
    {"name": "Padiyathalawa", "file": r"D:\...\Padiyathalawa_Welipitiya_RF_1990_2015.xlsx"}
]

sf_stations = [
    {"name": "Deraniyagala",  "file": r"D:\...\Deraniyagala_SF_1990-2015.xlsx"},
    {"name": "Nawalapitiya",  "file": r"D:\...\Nawalapitiya_SF_1990-2015.xlsx"},
    {"name": "Wellawaya",     "file": r"D:\...\Welawaya_SF_1990_2015.xlsx"},
    {"name": "Padiyathalawa", "file": r"D:\...\Padiyathalawa_SF_1990_2015.xlsx"}
]

seasons = {
    'FIM (Mar-Apr)': (3, 4),
    'SWM (May-Sep)': (5, 9),
    'SIM (Oct-Nov)': (10, 11),
    'NEM (Dec-Feb)': (12, 2)
}
season_order = list(seasons.keys())

def get_season(m):
    for name, (start, end) in seasons.items():
        if start <= end:
            if start <= m <= end:
                return name
        else:
            if m >= start or m <= end:
                return name

# =============================================================================
# 1. SEASONAL RAINFALL BOXPLOTS (per station)
# =============================================================================
all_data = []
for s in rf_stations:
    df = pd.read_excel(s["file"], usecols=[0, 1], names=['Date', 'Rainfall'], header=0)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].apply(get_season)
    seasonal = df.groupby(['Year', 'Season'])['Rainfall'].sum().reset_index()
    seasonal['Station'] = s["name"]
    all_data.append(seasonal)

full_df = pd.concat(all_data, ignore_index=True)
full_df['Season'] = pd.Categorical(full_df['Season'], categories=season_order, ordered=True)

for station in full_df['Station'].unique():
    data = full_df[full_df['Station'] == station]
    plt.figure(figsize=(6.5, 4))
    sns.boxplot(
        data=data, x='Season', y='Rainfall', width=0.55, color='#4A90E2',
        boxprops={'edgecolor': 'black', 'linewidth': 1.2},
        whiskerprops={'linewidth': 1.2}, capprops={'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 1.6}, fliersize=3,
        showmeans=True, meanline=True, meanprops={'color': 'red', 'ls': '-', 'lw': 1.6}
    )
    plt.ylim(0, 3500)
    plt.grid(axis='both', linestyle='-', alpha=0.6)
    plt.xlabel('Season', fontsize=14)
    plt.ylabel('Seasonal Cumulative Rainfall (mm)', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    filename = f"Seasonal_Rainfall_{station.replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 2. MEAN MONTHLY RAINFALL — per station bar chart
# =============================================================================
data = {}
for s in rf_stations:
    df = pd.read_excel(s["file"], usecols=[0, 1], names=['Date', 'Rainfall'], header=0, parse_dates=['Date'])
    data[s["name"]] = df.set_index('Date')['Rainfall'].sort_index()

def plot_monthly(station, series, ylabel, ylim, color, prefix):
    monthly = series.resample('ME').sum()
    monthly_df = monthly.to_frame()
    monthly_df['month'] = monthly_df.index.month
    means = monthly_df.groupby('month').mean().reindex(range(1, 13), fill_value=0).values.flatten()

    plt.figure(figsize=(6, 4))
    plt.bar(range(1, 13), means, color=color, edgecolor='black', alpha=0.9)
    plt.plot(range(1, 13), means, 'k-', lw=1.5)
    plt.ylabel(ylabel)
    plt.xlabel('Month')
    plt.xticks(range(1, 13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{station}.png"), dpi=400, bbox_inches='tight')
    plt.show()

for name, series in data.items():
    plot_monthly(name, series, 'Rainfall (mm/month)', (0, 500), 'blue', 'Mean_Monthly_Rainfall')

# =============================================================================
# 3. ANNUAL RAINFALL TRENDS — per station
# =============================================================================
data = {}
for st in rf_stations:
    path = st["file"]
    if not os.path.exists(path):
        print(f"WARNING: File not found -> {path}")
        continue
    df = pd.read_excel(path, parse_dates=['Date'])
    data[st["name"]] = df.set_index('Date')['Rainfall'].sort_index()

active_names = list(data.keys())
years_full = np.arange(1990, 2016)

for name in active_names:
    df = data[name]
    annual = df.resample('YE').sum()
    index_full = pd.date_range('1990-01-01', '2015-12-31', freq='YE')
    values = annual.reindex(index_full).values

    valid_mask = ~np.isnan(values)
    valid_years = years_full[valid_mask]
    valid_values = values[valid_mask]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(years_full, values, width=0.75, color='blue', edgecolor='black', linewidth=0.9, alpha=0.92, label=name)

    if len(valid_years) >= 26:
        slope, intercept, r_value, _, _ = linregress(valid_years, valid_values)
        trend_line = slope * years_full + intercept
        ax.plot(years_full, trend_line, color='black', linewidth=2.8, linestyle='--')

    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Annual Rainfall (mm/year)', fontsize=14)
    ax.set_xticks(years_full[::5])
    ax.set_xticklabels(years_full[::5], fontsize=12)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='both', linestyle='-', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(0, 7000)
    plt.tight_layout()

    safe_name = name.replace(" ", "_")
    filename = f"Annual_Rainfall_{safe_name}_1990-2015.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=400, bbox_inches='tight')
    plt.show()

# =============================================================================
# 4. SEASONAL STREAMFLOW BOXPLOTS (per station)
# =============================================================================
all_data = []
for s in sf_stations:
    df = pd.read_excel(s["file"], usecols=[0, 1], names=['Date', 'Discharge'], header=0)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].apply(get_season)
    seasonal = df.groupby(['Year', 'Season'])['Discharge'].sum().reset_index()
    seasonal['Station'] = s["name"]
    all_data.append(seasonal)

full_df_sf = pd.concat(all_data, ignore_index=True)
full_df_sf['Season'] = pd.Categorical(full_df_sf['Season'], categories=season_order, ordered=True)

for station in full_df_sf['Station'].unique():
    data_s = full_df_sf[full_df_sf['Station'] == station]
    plt.figure(figsize=(6.5, 4))
    sns.boxplot(
        data=data_s, x='Season', y='Discharge', width=0.55, color='#4A90E2',
        boxprops={'edgecolor': 'black', 'linewidth': 1.2},
        whiskerprops={'linewidth': 1.2}, capprops={'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 1.6}, fliersize=3,
        showmeans=True, meanline=True, meanprops={'color': 'red', 'ls': '-', 'lw': 1.6}
    )
    plt.ylim(0, 4000)
    plt.grid(axis='both', linestyle='-', alpha=0.6)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Seasonal Cumulative Discharge (m$^3$/sec)', fontsize=12)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    filename = f"Seasonal_streamflow_{station.replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 5. MEAN MONTHLY STREAMFLOW — per station bar chart
# =============================================================================
data = {}
for s in sf_stations:
    df = pd.read_excel(s["file"], usecols=[0, 1], names=['Date', 'Discharge'], header=0, parse_dates=['Date'])
    data[s["name"]] = df.set_index('Date')['Discharge'].sort_index()

for name, series in data.items():
    plot_monthly(name, series, r'Monthly Streamflow (m$^3$/sec)', (0, 900), 'skyblue', 'Mean_Monthly_streamflow')

# =============================================================================
# 6. COMBINED 2x2 MEAN MONTHLY RAINFALL (all stations)
# =============================================================================
data = {}
for s in rf_stations:
    df = pd.read_excel(s["file"], usecols=[0, 1], names=['Date', 'Rainfall'], header=0, parse_dates=['Date'])
    data[s["name"]] = df.set_index('Date')['Rainfall'].sort_index()

fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharey=True)
axes = axes.flatten()
panel_labels = ['(a)', '(b)', '(c)', '(d)']
months = range(1, 13)
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

for ax, (station, series), label in zip(axes, data.items(), panel_labels):
    monthly = series.resample('ME').sum()
    monthly_df = monthly.to_frame()
    monthly_df['month'] = monthly_df.index.month
    means = monthly_df.groupby('month').mean().reindex(months, fill_value=0).values.flatten()

    ax.bar(months, means, color='blue', edgecolor='black', alpha=0.9)
    ax.plot(months, means, 'k-', lw=1.5)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylim(0, 550)
    ax.tick_params(axis='y', labelleft=True)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.98, f"{label} {station}", transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

fig.text(0.5, 0.04, 'Month', ha='center', fontsize=12, fontweight='bold')
fig.text(0.04, 0.5, 'Rainfall (mm/month)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0.06, 0.06, 1, 1])
plt.savefig(os.path.join(output_dir, "Mean_Monthly_Rainfall_All_Stations.png"), dpi=400, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. COMBINED 2x2 ANNUAL RAINFALL (all stations)
# =============================================================================
data = {}
for st in rf_stations:
    df = pd.read_excel(st["file"], parse_dates=['Date'])
    data[st["name"]] = df.set_index('Date')['Rainfall'].sort_index()

years_full = np.arange(1990, 2016)
index_full = pd.date_range('1990-01-01', '2015-12-31', freq='YE')
panel_labels = ['(a)', '(b)', '(c)', '(d)']

fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharey=True)
axes = axes.flatten()

for ax, (name, series), label in zip(axes, data.items(), panel_labels):
    annual = series.resample('YE').sum()
    values = annual.reindex(index_full).values.flatten()
    valid_mask = ~np.isnan(values)
    valid_years = years_full[valid_mask]
    valid_values = values[valid_mask]

    ax.bar(years_full, values, width=0.75, color='steelblue', edgecolor='black', linewidth=0.9, alpha=0.92)

    if len(valid_years) >= 26:
        slope, intercept, _, _, _ = linregress(valid_years, valid_values)
        ax.plot(years_full, slope * years_full + intercept, color='black', linewidth=2.2, linestyle='--')

    ax.set_xticks(years_full[::5])
    ax.set_xticklabels(years_full[::5])
    ax.set_ylim(0, 7000)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis='y', labelleft=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.95, f"{label} {name}", transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

fig.text(0.5, 0.04, 'Year', ha='center', fontsize=13)
fig.text(0.04, 0.5, 'Annual Rainfall (mm/year)', va='center', rotation='vertical', fontsize=13)
plt.tight_layout(rect=[0.06, 0.06, 1, 1])
out_path = os.path.join(output_dir, "Annual_Rainfall_All_Stations_1990-2015.png")
plt.savefig(out_path, dpi=400, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. COMBINED 2x2 MEAN MONTHLY STREAMFLOW (all stations)
# =============================================================================
data = {}
for s in sf_stations:
    df = pd.read_excel(s["file"], usecols=[0, 1], names=['Date', 'Discharge'], header=0, parse_dates=['Date'])
    data[s["name"]] = df.set_index('Date')['Discharge'].sort_index()

fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharey=True)
axes = axes.flatten()

for ax, (name, series), label in zip(axes, data.items(), panel_labels):
    monthly = series.resample('ME').sum()
    monthly_df = monthly.to_frame()
    monthly_df['month'] = monthly_df.index.month
    means = monthly_df.groupby('month').mean().reindex(months, fill_value=0).values.flatten()

    ax.bar(months, means, color='skyblue', edgecolor='black', alpha=0.9)
    ax.plot(months, means, 'k-', lw=1.5)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylim(0, 900)
    ax.tick_params(axis='y', labelleft=True)
    ax.text(0.02, 0.98, f"{label} {name}", transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

fig.text(0.5, 0.04, 'Month', ha='center', fontsize=13)
fig.text(0.04, 0.5, r'Monthly Streamflow (m$^3$/sec)', va='center', rotation='vertical', fontsize=13)
plt.tight_layout(rect=[0.06, 0.06, 1, 1])
out_file = os.path.join(output_dir, "Mean_Monthly_Streamflow_All_Stations.png")
plt.savefig(out_file, dpi=400, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. COMBINED SEASONAL STREAMFLOW COMPARISON (grouped boxplot, all stations)
# =============================================================================
station_colors = {s["name"]: c for s, c in zip(sf_stations, ['#4A90E2', '#50C878', '#F5A623', '#D0021B'])}

plt.figure(figsize=(7, 4))
n_stations = len(station_colors)
width = 0.18

for idx, station_name in enumerate(full_df_sf['Station'].unique()):
    data_s = full_df_sf[full_df_sf['Station'] == station_name]
    positions = [i + idx * width for i in range(len(season_order))]
    plt.boxplot(
        [data_s[data_s['Season'] == season]['Discharge'] for season in season_order],
        positions=positions, widths=width, patch_artist=True,
        boxprops=dict(facecolor=station_colors[station_name], color='black', linewidth=1.2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        medianprops=dict(color='black', linewidth=1.6),
        flierprops=dict(marker='o', markersize=3, markerfacecolor='grey'),
        meanline=True, showmeans=True, meanprops=dict(color='red', linewidth=1.6)
    )

tick_positions = [i + width * (n_stations - 1) / 2 for i in range(len(season_order))]
plt.xticks(tick_positions, season_order, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Season', fontsize=13)
plt.ylabel('Seasonal Cumulative Discharge (m$^3$/sec)', fontsize=13)
plt.ylim(0, 4000)
plt.grid(axis='both', linestyle='-', alpha=0.6)

legend_handles = [Patch(facecolor=color, edgecolor='black', label=name) for name, color in station_colors.items()]
plt.legend(handles=legend_handles, ncol=2, loc='upper right', fontsize=12)
plt.title('Seasonal Streamflow Comparison across all stations (1990-2015)', fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.12)

out_file = os.path.join(output_dir, "Seasonal_Streamflow_All_Stations_Combined.png")
plt.savefig(out_file, dpi=300, bbox_inches=None)
plt.show()
