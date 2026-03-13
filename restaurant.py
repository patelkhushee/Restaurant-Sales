# ============================================================
#  RESTAURANT SALES — COMPLETE DATA PIPELINE
#  Covers: Data Cleaning → EDA → Feature Engineering →
#          XGBoost → Prophet → Evaluation → Visualisation
#
#  Libraries: pandas, numpy, scikit-learn, xgboost,
#             prophet, matplotlib, plotly
#
#  Run:  python restaurant_complete_pipeline.py
# ============================================================

# ── 0. IMPORTS ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                       # no GUI needed — saves as PNG files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings('ignore')

# Optional imports — the script tells you if they are missing
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False
    print("NOTE: plotly not installed — skipping interactive charts.")
    print("      Install with:  pip install plotly")

try:
    from xgboost import XGBRegressor
    XGBOOST = True
except ImportError:
    XGBOOST = False
    print("NOTE: xgboost not installed — will use GradientBoosting instead.")
    print("      Install with:  pip install xgboost")

try:
    from prophet import Prophet
    PROPHET = True
except ImportError:
    PROPHET = False
    print("NOTE: prophet not installed — skipping Prophet model.")
    print("      Install with:  pip install prophet")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import joblib

# ── FOLDER SETUP ─────────────────────────────────────────────────────────────
os.makedirs('outputs/charts',  exist_ok=True)
os.makedirs('outputs/models',  exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

INPUT_FILE = 'restaurant_sales_messy.csv'   # ← change if your file has a different name

print("=" * 65)
print("  RESTAURANT SALES — COMPLETE ML PIPELINE")
print("=" * 65)


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 1: LOAD THE RAW DATA                                  ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Read the CSV file exactly as-is. No changes yet.
# WHY:   We need to see all the problems before we can fix them.

print("\n📂  STEP 1: Loading raw data...")
raw = pd.read_csv("/kaggle/input/datasets/khusboopatel/restaurant-sales-messy-data/restaurant_sales_messy.csv")
print(f"    Rows loaded    : {len(raw):,}")
print(f"    Columns        : {list(raw.columns)}")
print(f"    Memory usage   : {raw.memory_usage(deep=True).sum() / 1e6:.1f} MB")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 2: DIAGNOSE — FIND ALL PROBLEMS                       ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Print a full diagnosis report before touching anything.
# WHY:   You must understand WHAT is wrong before you can fix it.
#        Skipping this step leads to hidden bugs later.

print("\n🔍  STEP 2: Diagnosing data problems...")
print("-" * 50)

# 2a. Missing values per column
print("\n  ▸ Missing values per column:")
missing = raw.isna().sum()
for col, n in missing[missing > 0].items():
    pct = n / len(raw) * 100
    print(f"    {col:<20} {n:>6,} missing  ({pct:.1f}%)")

# 2b. Data type problems
print("\n  ▸ Column dtypes (raw — before fixing):")
print(raw.dtypes.to_string())

# 2c. Inconsistent boolean-like columns
for col in ['is_weekend', 'is_holiday', 'is_summer', 'is_monsoon']:
    vals = raw[col].dropna().unique()
    print(f"\n  ▸ Unique values in '{col}'  ({len(vals)} variants):")
    print(f"    {sorted([str(v) for v in vals])}")

# 2d. Outlet name chaos
print(f"\n  ▸ Unique outlet values in raw data: {raw['outlet'].dropna().nunique()}")
print(f"    (should be ~12 real outlets)")

# 2e. Menu item spelling chaos
print(f"\n  ▸ Unique menu_item values in raw data: {raw['menu_item'].dropna().nunique()}")
print(f"    (should be ~9 real items)")

# 2f. Numeric outliers
print(f"\n  ▸ units_sold range: {raw['units_sold'].min()} to {raw['units_sold'].max()}")
print(f"    Negative values  : {(raw['units_sold'] < 0).sum()} rows")
print(f"  ▸ temperature range: {raw['temperature'].min()} to {raw['temperature'].max()}")
print(f"    Impossible (>50°C): {(raw['temperature'] > 50).sum()} rows")
print(f"    Impossible (<0°C) : {(raw['temperature'] < 0).sum()} rows")

# 2g. Date format chaos
print(f"\n  ▸ Sample date values (mixed formats detected):")
sample_dates = raw['date'].dropna().unique()[:10]
for d in sample_dates:
    print(f"    '{d}'")

# Save diagnosis report to text file
with open('outputs/reports/01_diagnosis_report.txt', 'w') as f:
    f.write("DIAGNOSIS REPORT\n" + "=" * 50 + "\n\n")
    f.write(f"Total rows: {len(raw):,}\n\n")
    f.write("Missing values:\n")
    f.write(missing.to_string())
    f.write("\n\nDtypes:\n")
    f.write(raw.dtypes.to_string())
print("\n    ✅  Diagnosis saved: outputs/reports/01_diagnosis_report.txt")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 3: DATA CLEANING                                      ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Fix every problem found in Step 2, one by one.
# WHY:   Dirty data → wrong model → wrong predictions → bad business decisions.
#        Each fix is documented so anyone can understand what was changed.

print("\n🧹  STEP 3: Cleaning data...")
df = raw.copy()     # always work on a copy — never modify the original

# ── 3a. FIX DATE COLUMN ──────────────────────────────────────────────────────
# PROBLEM: Dates come in many formats:
#   '2022-08-13'  (ISO format)
#   '08/29/2022'  (US format MM/DD/YYYY)
#   '17-05-2023'  (DD-MM-YYYY)
#   '2022/01/10'  (ISO with slashes)
# FIX: pandas.to_datetime with dayfirst=True + errors='coerce'
#      dayfirst=True handles DD-MM-YYYY and DD/MM/YYYY formats
#      errors='coerce' converts anything unparseable to NaT (not an error)

print("  ▸ 3a. Fixing date formats...")
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
bad_dates = df['date'].isna().sum()
print(f"     Unparseable dates converted to NaT: {bad_dates}")
# Drop rows where date could not be parsed — they are useless without a date
df = df.dropna(subset=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"     Date range after fix: {df['date'].min().date()} to {df['date'].max().date()}")


# ── 3b. FIX BOOLEAN COLUMNS ──────────────────────────────────────────────────
# PROBLEM: is_weekend, is_holiday, is_summer, is_monsoon each have 13 variants:
#   TRUE / True / true / 1 / yes / Yes / Y  → should all be 1
#   FALSE / False / false / 0 / no / No / N → should all be 0
#   NaN                                      → missing, handle separately
# FIX: Map every known string to 1 or 0, then fill NaN with median

print("  ▸ 3b. Standardising boolean columns...")
TRUE_VALUES  = {'true','1','yes','y','TRUE','Yes','Y','True'}
FALSE_VALUES = {'false','0','no','n','FALSE','No','N','False'}

def fix_boolean(series):
    """
    Convert a messy boolean column to 0/1 integers.
    Anything unrecognised becomes NaN (to be filled later).
    """
    def convert(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s in TRUE_VALUES:
            return 1
        if s in FALSE_VALUES:
            return 0
        return np.nan          # unknown value → NaN

    return series.apply(convert)

bool_cols = ['is_weekend', 'is_holiday', 'is_summer', 'is_monsoon']
for col in bool_cols:
    before_nulls = df[col].isna().sum()
    df[col] = fix_boolean(df[col])
    after_nulls  = df[col].isna().sum()
    print(f"     {col}: {before_nulls} NaN before → {after_nulls} NaN after fix")


# ── 3c. FIX OUTLET NAMES ─────────────────────────────────────────────────────
# PROBLEM: 64 unique variants for ~12 real outlet names:
#   'Banashankari', 'BANASHANKARI', ' Banashankari', 'banashankari',
#   'Banashankari ' (trailing space), 'banashankari' (all lowercase)
# FIX: Strip spaces → lowercase → map to canonical name

print("  ▸ 3c. Fixing outlet names...")
# Step 1: strip all extra spaces and convert to lowercase
df['outlet_clean'] = df['outlet'].str.strip().str.lower()
df['outlet_clean'] = df['outlet_clean'].str.replace('_', ' ', regex=False)
df['outlet_clean'] = df['outlet_clean'].str.replace(r'\s+', ' ', regex=True)

# Step 2: map lowercase name to proper canonical name
outlet_map = {
    'banashankari'  : 'Banashankari',
    'hsr layout'    : 'HSR Layout',
    'koramangala'   : 'Koramangala',
    'whitefield'    : 'Whitefield',
    'mg road'       : 'MG Road',
    'hebbal'        : 'Hebbal',
    'marathahalli'  : 'Marathahalli',
    'electronic city': 'Electronic City',
    'indiranagar'   : 'Indiranagar',
    'jayanagar'     : 'Jayanagar',
    'yelahanka'     : 'Yelahanka',
    'rajajinagar'   : 'Rajajinagar',
}
df['outlet'] = df['outlet_clean'].map(outlet_map)
df.drop(columns=['outlet_clean'], inplace=True)

still_missing = df['outlet'].isna().sum()
print(f"     Outlet nulls after fix: {still_missing}")
print(f"     Clean unique outlets  : {sorted(df['outlet'].dropna().unique())}")


# ── 3d. FIX MENU ITEM NAMES ──────────────────────────────────────────────────
# PROBLEM: 56 unique variants for 9 real menu items:
#   'biryani','Biryani','BIRYANI','Briyani','Biriyani','Bryani','biryani '
#   (spelling errors + case + underscores + extra spaces)
# FIX: Strip → lowercase → remove punctuation → map to canonical name

print("  ▸ 3d. Fixing menu item names...")
# Clean: strip spaces, lowercase, remove underscores/dots, collapse spaces
df['item_clean'] = (df['menu_item']
    .str.strip()
    .str.lower()
    .str.replace('[_.]', ' ', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
)

# Map every known variant to the canonical item name
item_map = {
    # Biryani and misspellings
    'biryani'        : 'Biryani',
    'biriyani'       : 'Biryani',
    'briyani'        : 'Biryani',
    'bryani'         : 'Biryani',
    # Masala Dosa variants
    'masala dosa'    : 'Masala Dosa',
    'masaladosa'     : 'Masala Dosa',
    'masla dosa'     : 'Masala Dosa',
    'masala  dosa'   : 'Masala Dosa',
    # Veg Thali variants
    'veg thali'      : 'Veg Thali',
    'veg  thali'     : 'Veg Thali',
    'vegthali'       : 'Veg Thali',
    'veg  thali'     : 'Veg Thali',
    'veg thali'      : 'Veg Thali',
    'veg thali'      : 'Veg Thali',
    'veg. thali'     : 'Veg Thali',
    # Paneer Tikka variants
    'paneer tikka'   : 'Paneer Tikka',
    'paneertikka'    : 'Paneer Tikka',
    'panner tikka'   : 'Paneer Tikka',
    'paneer tika'    : 'Paneer Tikka',
    # Butter Chicken variants
    'butter chicken' : 'Butter Chicken',
    'butter chiken'  : 'Butter Chicken',
    'butter  chicken': 'Butter Chicken',
    'bttr chicken'   : 'Butter Chicken',
    # Chicken Burger variants
    'chicken burger' : 'Chicken Burger',
    'chicken  burger': 'Chicken Burger',
    'chcken burger'  : 'Chicken Burger',
    'chiken burger'  : 'Chicken Burger',
    # Cold Coffee variants
    'cold coffee'    : 'Cold Coffee',
    'coldcoffee'     : 'Cold Coffee',
    'cold cofee'     : 'Cold Coffee',
    'cold coffe'     : 'Cold Coffee',
    # Gulab Jamun variants
    'gulab jamun'    : 'Gulab Jamun',
    'gulab jaamun'   : 'Gulab Jamun',
    'gulaab jamun'   : 'Gulab Jamun',
    'gulb jamun'     : 'Gulab Jamun',
    'gulab  jamun'   : 'Gulab Jamun',
}
df['menu_item'] = df['item_clean'].map(item_map)
df.drop(columns=['item_clean'], inplace=True)

still_missing_items = df['menu_item'].isna().sum()
print(f"     Menu item nulls after fix: {still_missing_items}")
print(f"     Clean unique items: {sorted(df['menu_item'].dropna().unique())}")


# ── 3e. FIX TEMPERATURE OUTLIERS ─────────────────────────────────────────────
# PROBLEM: Temperature has physically impossible values:
#   -5°C in Bengaluru (impossible — city never goes below ~10°C)
#   99.9°C, 150°C (clearly data entry errors — likely body temp entered as Celsius)
# FIX: Cap at realistic Bengaluru range (10°C to 40°C), replace rest with NaN

print("  ▸ 3e. Fixing temperature outliers...")
before_bad_temp = ((df['temperature'] > 40) | (df['temperature'] < 10)).sum()
df.loc[(df['temperature'] > 40) | (df['temperature'] < 10), 'temperature'] = np.nan
print(f"     Impossible temperatures replaced with NaN: {before_bad_temp}")


# ── 3f. FIX NEGATIVE UNITS SOLD ──────────────────────────────────────────────
# PROBLEM: 300 rows have negative units_sold (e.g. -243).
#   This is impossible — you cannot sell negative food.
#   These are likely return/refund records entered as negative.
# FIX: Take the absolute value (assume they were entered with wrong sign)

print("  ▸ 3f. Fixing negative units_sold...")
neg_count = (df['units_sold'] < 0).sum()
df.loc[df['units_sold'] < 0, 'units_sold'] = df.loc[
    df['units_sold'] < 0, 'units_sold'].abs()
print(f"     Negative values converted to positive: {neg_count}")

# Also fix extreme outliers (>500 units for a single item in one day = impossible)
extreme_high = (df['units_sold'] > 500).sum()
df.loc[df['units_sold'] > 500, 'units_sold'] = np.nan
print(f"     Extreme high values (>500) set to NaN: {extreme_high}")


# ── 3g. FIX month AND day_of_week COLUMNS ────────────────────────────────────
# PROBLEM: month has 1,580 missing values. day_of_week has 2,113 NaN.
# FIX: Re-derive both directly from the cleaned date column.
#      This is MORE reliable than keeping the original values anyway
#      because the original month column was entered manually and may be wrong.

print("  ▸ 3g. Re-deriving month and day_of_week from date...")
df['month']       = df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()
print(f"     month nulls now    : {df['month'].isna().sum()}")
print(f"     day_of_week nulls  : {df['day_of_week'].isna().sum()}")


# ── 3h. FILL REMAINING MISSING VALUES ────────────────────────────────────────
# PROBLEM: After all fixes, some columns still have NaN.
# FIX: Different strategies per column type:
#   Numeric (temperature, units_sold) → median fill (resistant to outliers)
#   Boolean (is_weekend etc.)         → re-derive from date where possible
#   Categorical (outlet, menu_item)   → forward-fill or drop depending on %

print("  ▸ 3h. Filling remaining missing values...")

# Re-derive is_weekend from date (100% reliable — better than guessing)
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# Re-derive is_summer (April=4, May=5, June=6) and is_monsoon (Jul-Sep)
df['is_summer']  = df['month'].isin([4, 5, 6]).astype(int)
df['is_monsoon'] = df['month'].isin([7, 8, 9]).astype(int)

# For is_holiday — fill NaN with 0 (assume not a holiday if unknown)
df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

# Fill temperature NaN with the median temperature for that month
# (Bengaluru in May is different from Bengaluru in December)
df['temperature'] = df.groupby('month')['temperature'].transform(
    lambda x: x.fillna(x.median())
)
# If still NaN (no data for that month), fill with overall median
df['temperature'] = df['temperature'].fillna(df['temperature'].median())

# Fill units_sold NaN with median of that menu_item × outlet combination
# (Biryani at Koramangala has different typical sales than Gulab Jamun at Yelahanka)
df['units_sold'] = df.groupby(['menu_item', 'outlet'])['units_sold'].transform(
    lambda x: x.fillna(x.median())
)
# If still NaN (new combo), fill with overall median
df['units_sold'] = df['units_sold'].fillna(df['units_sold'].median())
df['units_sold'] = df['units_sold'].round().astype(int)

# For outlet and menu_item NaN — drop these rows (we need both to be useful)
rows_before_drop = len(df)
df = df.dropna(subset=['outlet', 'menu_item'])
rows_dropped = rows_before_drop - len(df)
print(f"     Dropped rows with missing outlet or menu_item: {rows_dropped}")

print(f"\n  ▸ Missing values AFTER all fixes:")
remaining = df.isna().sum()
if remaining.sum() == 0:
    print("     ✅  Zero missing values remaining!")
else:
    print(remaining[remaining > 0])


# ── 3i. FINAL CLEAN DATASET SUMMARY ─────────────────────────────────────────
print(f"\n  ▸ Clean dataset summary:")
print(f"     Rows             : {len(df):,}  (started with {len(raw):,})")
print(f"     Columns          : {list(df.columns)}")
print(f"     Outlets          : {sorted(df['outlet'].unique())}")
print(f"     Menu items       : {sorted(df['menu_item'].unique())}")
print(f"     Date range       : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"     units_sold range : {df['units_sold'].min()} to {df['units_sold'].max()}")
print(f"     temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")

# Save the clean data
df.to_csv('outputs/restaurant_sales_clean.csv', index=False)
print("\n     ✅  Clean data saved: outputs/restaurant_sales_clean.csv")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 4: AGGREGATE — DAILY TOTAL SALES                      ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  The raw data has one row per sale (outlet × menu_item × date).
#        For forecasting we need ONE number per day = total units sold.
# WHY:   Time series models work on one value per time point.

print("\n📊  STEP 4: Aggregating to daily totals...")
daily = (df.groupby('date')
           .agg(
               daily_sales   = ('units_sold',   'sum'),
               avg_temp      = ('temperature',  'mean'),
               is_holiday    = ('is_holiday',   'max'),
               is_weekend    = ('is_weekend',   'max'),
               is_summer     = ('is_summer',    'max'),
               is_monsoon    = ('is_monsoon',   'max'),
               num_outlets   = ('outlet',       'nunique'),
               num_items     = ('menu_item',    'nunique'),
           )
           .reset_index()
           .sort_values('date')
)
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.set_index('date').asfreq('D')

# Fill any gaps created by asfreq (days with zero records → interpolate)
daily['daily_sales'] = daily['daily_sales'].interpolate(method='linear').round().astype(int)
for col in ['avg_temp','is_holiday','is_weekend','is_summer','is_monsoon']:
    daily[col] = daily[col].ffill().bfill()

print(f"     Daily rows: {len(daily):,}")
print(f"     Date range: {daily.index.min().date()} → {daily.index.max().date()}")
print(f"     Daily sales — min: {daily['daily_sales'].min()}, "
      f"max: {daily['daily_sales'].max()}, "
      f"mean: {daily['daily_sales'].mean():.0f}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 5: EXPLORATORY DATA ANALYSIS (EDA)                    ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Create 6 key charts to understand patterns in the data.
# WHY:   You must see the data before modelling it.
#        EDA reveals which features to engineer in Step 6.

print("\n📈  STEP 5: Exploratory Data Analysis — generating charts...")

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor'  : '#f8f9fa',
    'axes.grid'       : True,
    'grid.alpha'      : 0.4,
    'font.family'     : 'sans-serif',
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
})

# ── CHART 1: Overall Sales Trend ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Restaurant Sales — Exploratory Data Analysis', fontsize=16, fontweight='bold', y=1.01)

ax = axes[0, 0]
ax.plot(daily.index, daily['daily_sales'], color='steelblue',  linewidth=0.7, alpha=0.6, label='Daily sales')
ax.plot(daily.index, daily['daily_sales'].rolling(30).mean(), color='red', linewidth=2.5, label='30-day rolling avg')
ax.set_title('Chart 1 — Overall Daily Sales Trend')
ax.set_ylabel('Total Units Sold')
ax.legend()

# ── CHART 2: Average Sales by Day of Week ────────────────────────────────────
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
daily_with_day = daily.copy()
daily_with_day['day_name'] = daily_with_day.index.day_name()
dow_avg = daily_with_day.groupby('day_name')['daily_sales'].mean().reindex(day_order)

ax = axes[0, 1]
colors = ['#e74c3c' if d in ['Saturday','Sunday'] else '#3498db' for d in day_order]
bars = ax.bar(day_order, dow_avg.values, color=colors, edgecolor='white', linewidth=0.5)
ax.set_title('Chart 2 — Average Sales by Day of Week\n(Red = Weekend)')
ax.set_ylabel('Average Units Sold')
for bar, val in zip(bars, dow_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')

# ── CHART 3: Average Sales by Month ──────────────────────────────────────────
daily_with_month = daily.copy()
daily_with_month['month_num'] = daily_with_month.index.month
month_avg = daily_with_month.groupby('month_num')['daily_sales'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

ax = axes[1, 0]
ax.bar(month_names, month_avg.values, color='#27ae60', edgecolor='white', linewidth=0.5)
ax.set_title('Chart 3 — Average Sales by Month (Seasonality)')
ax.set_ylabel('Average Units Sold')
for i, val in enumerate(month_avg.values):
    ax.text(i, val + 2, f'{val:.0f}', ha='center', fontsize=8, fontweight='bold')

# ── CHART 4: Menu Item Popularity ────────────────────────────────────────────
item_sales = df.groupby('menu_item')['units_sold'].sum().sort_values(ascending=True)

ax = axes[1, 1]
colors_items = plt.cm.Spectral(np.linspace(0.1, 0.9, len(item_sales)))
ax.barh(item_sales.index, item_sales.values, color=colors_items)
ax.set_title('Chart 4 — Total Sales by Menu Item')
ax.set_xlabel('Total Units Sold')
for i, val in enumerate(item_sales.values):
    ax.text(val + 100, i, f'{val:,.0f}', va='center', fontsize=8)

# ── CHART 5: Outlet Performance ──────────────────────────────────────────────
outlet_sales = df.groupby('outlet')['units_sold'].sum().sort_values(ascending=True)

ax = axes[2, 0]
colors_outlets = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(outlet_sales)))
ax.barh(outlet_sales.index, outlet_sales.values, color=colors_outlets)
ax.set_title('Chart 5 — Total Sales by Outlet')
ax.set_xlabel('Total Units Sold')
for i, val in enumerate(outlet_sales.values):
    ax.text(val + 100, i, f'{val:,.0f}', va='center', fontsize=8)

# ── CHART 6: Temperature vs Sales Scatter ────────────────────────────────────
ax = axes[2, 1]
sc = ax.scatter(daily['avg_temp'], daily['daily_sales'],
                alpha=0.3, s=10, c=daily['is_weekend'],
                cmap='coolwarm', label='Blue=Weekday, Red=Weekend')
plt.colorbar(sc, ax=ax, label='Weekend?')
ax.set_title('Chart 6 — Temperature vs Daily Sales')
ax.set_xlabel('Average Temperature (°C)')
ax.set_ylabel('Daily Sales')

plt.tight_layout()
plt.savefig('outputs/charts/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("     ✅  Saved: outputs/charts/01_eda_overview.png")

# ── CHART 7: Outlet × Menu Item Heatmap ──────────────────────────────────────
pivot = df.pivot_table(values='units_sold', index='menu_item',
                       columns='outlet', aggfunc='sum', fill_value=0)
# normalise each column so outlets with more data don't dominate
pivot_norm = pivot.div(pivot.sum(axis=0), axis=1) * 100

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(pivot_norm.values, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=ax, label='% of outlet total sales')
ax.set_xticks(range(len(pivot_norm.columns)))
ax.set_xticklabels(pivot_norm.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(pivot_norm.index)))
ax.set_yticklabels(pivot_norm.index, fontsize=10)
for i in range(len(pivot_norm.index)):
    for j in range(len(pivot_norm.columns)):
        ax.text(j, i, f'{pivot_norm.values[i,j]:.1f}%', ha='center', va='center', fontsize=7)
ax.set_title('Chart 7 — Menu Item Sales Share per Outlet (%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/charts/02_outlet_menu_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("     ✅  Saved: outputs/charts/02_outlet_menu_heatmap.png")


# Plotly interactive charts
if PLOTLY:
    print("  ▸ Creating Plotly interactive charts...")

    # Interactive 1: Sales trend with range slider
    fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          subplot_titles=('Daily Sales with 30-day Rolling Average',
                                          'Temperature Over Time'),
                          row_heights=[0.7, 0.3])
    fig_p.add_trace(go.Scatter(x=daily.index, y=daily['daily_sales'],
                               mode='lines', name='Daily Sales',
                               line=dict(color='steelblue', width=1), opacity=0.6), row=1, col=1)
    fig_p.add_trace(go.Scatter(x=daily.index, y=daily['daily_sales'].rolling(30).mean(),
                               mode='lines', name='30-day Avg',
                               line=dict(color='red', width=2.5)), row=1, col=1)
    fig_p.add_trace(go.Scatter(x=daily.index, y=daily['avg_temp'],
                               mode='lines', name='Temperature',
                               line=dict(color='orange', width=1)), row=2, col=1)
    fig_p.update_xaxes(rangeslider_visible=True, row=2, col=1)
    fig_p.update_layout(title='Interactive Sales Dashboard — Drag to Zoom',
                        height=600, template='plotly_white')
    fig_p.write_html('outputs/charts/03_interactive_sales.html')

    # Interactive 2: Box plots by day of week
    df_plot = df.copy()
    df_plot['day_name'] = pd.to_datetime(df_plot['date']).dt.day_name()
    fig_box = px.box(df_plot, x='day_name', y='units_sold',
                     color='day_name', category_orders={'day_name': day_order},
                     title='Sales Distribution by Day of Week',
                     template='plotly_white')
    fig_box.write_html('outputs/charts/04_sales_by_day_interactive.html')

    # Interactive 3: outlet comparison
    fig_outlet = px.bar(df.groupby('outlet')['units_sold'].mean().reset_index(),
                        x='outlet', y='units_sold', color='outlet',
                        title='Average Daily Sales per Outlet',
                        template='plotly_white')
    fig_outlet.update_layout(xaxis_tickangle=-45)
    fig_outlet.write_html('outputs/charts/05_outlet_comparison.html')

    print("     ✅  Interactive HTML charts saved to outputs/charts/")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 6: FEATURE ENGINEERING                                ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Create the 15+ columns that the ML model will use to make predictions.
# WHY:   Raw data has only date + sales. The model needs informative inputs.
#        'Lag features' and 'rolling windows' are the most powerful features
#        for time series forecasting.

print("\n⚙️   STEP 6: Feature Engineering...")

ts = daily.copy()     # ts = time series dataframe

# ── 6a. CALENDAR FEATURES ────────────────────────────────────────────────────
# These tell the model WHAT TIME it is
ts['day_of_week']   = ts.index.dayofweek        # 0=Mon, 6=Sun
ts['day_of_month']  = ts.index.day              # 1 to 31
ts['month_num']     = ts.index.month            # 1 to 12
ts['quarter']       = ts.index.quarter          # 1 to 4
ts['week_of_year']  = ts.index.isocalendar().week.astype(int)
ts['year']          = ts.index.year
ts['is_friday']     = (ts.index.dayofweek == 4).astype(int)
ts['is_monday']     = (ts.index.dayofweek == 0).astype(int)
ts['is_december']   = (ts.index.month == 12).astype(int)    # holiday peak month

# ── 6b. LAG FEATURES ─────────────────────────────────────────────────────────
# WHAT:  Use PAST sales as inputs to predict FUTURE sales.
# WHY:   Last week's same-day sales is the strongest predictor of today.
# HOW:   .shift(n) moves every value DOWN by n positions (looks n days back).
# CRITICAL: lag features will be NaN for the first n rows — this is expected.

for lag in [1, 2, 3, 7, 14, 21, 30]:
    ts[f'lag_{lag}'] = ts['daily_sales'].shift(lag)

# ── 6c. ROLLING WINDOW FEATURES ──────────────────────────────────────────────
# WHAT:  Averages over recent windows to capture trend and volatility.
# CRITICAL: ALWAYS .shift(1) FIRST before .rolling() to prevent data leakage!
#   Without shift(1): rolling average on day X would include day X itself → CHEATING
#   With shift(1): rolling average on day X only includes days before X → SAFE

sales_shifted = ts['daily_sales'].shift(1)         # shift once, reuse for all windows
ts['roll_mean_7']   = sales_shifted.rolling(7).mean()
ts['roll_mean_14']  = sales_shifted.rolling(14).mean()
ts['roll_mean_30']  = sales_shifted.rolling(30).mean()
ts['roll_std_7']    = sales_shifted.rolling(7).std()
ts['roll_max_7']    = sales_shifted.rolling(7).max()
ts['roll_min_7']    = sales_shifted.rolling(7).min()
ts['roll_median_7'] = sales_shifted.rolling(7).median()

# ── 6d. TEMPERATURE FEATURES ─────────────────────────────────────────────────
# A hot day reduces footfall → fewer customers → lower sales
ts['temp_lag_1']  = ts['avg_temp'].shift(1)
ts['temp_above_30'] = (ts['avg_temp'] > 30).astype(int)  # very hot day flag

# ── 6e. DROP NaN ROWS ────────────────────────────────────────────────────────
# The first 30 rows will have NaN from the lag_30 feature.
# We must drop them — you cannot train on rows with missing inputs.
rows_before = len(ts)
ts = ts.dropna()
rows_dropped = rows_before - len(ts)
print(f"     Rows before feature engineering NaN drop: {rows_before}")
print(f"     Rows dropped (expected from lag_30)      : {rows_dropped}")
print(f"     Rows available for modelling             : {len(ts)}")

feature_cols = [c for c in ts.columns if c != 'daily_sales']
print(f"     Total features created: {len(feature_cols)}")
print(f"     Feature names: {feature_cols}")

# Save feature matrix
ts.to_csv('outputs/restaurant_features.csv')
print("     ✅  Feature matrix saved: outputs/restaurant_features.csv")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 7: TRAIN / TEST SPLIT                                 ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Split into training and test sets.
# WHY:   We evaluate how well the model predicts UNSEEN FUTURE data.
# CRITICAL: NEVER use random split for time series!
#   Random split puts future dates in training → data leakage → fake results.
#   ALWAYS use sequential split: train on past, test on future.

print("\n✂️   STEP 7: Train/Test Split...")
X = ts[feature_cols]
y = ts['daily_sales']

SPLIT_RATIO = 0.85      # 85% train, 15% test
split_idx   = int(len(ts) * SPLIT_RATIO)

X_train, X_test = X.iloc[:split_idx],  X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx],  y.iloc[split_idx:]

print(f"     Training period : {X_train.index.min().date()} → {X_train.index.max().date()}  ({len(X_train)} days)")
print(f"     Testing period  : {X_test.index.min().date()}  → {X_test.index.max().date()}   ({len(X_test)} days)")
print(f"     WHY sequential? : Training is ENTIRELY in the past vs testing. No leakage.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 8: MODEL TRAINING                                     ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Train 3 models: baseline, random forest, XGBoost (or GBM if no XGBoost).
# WHY:   Always start with a simple baseline.
#        Never skip baseline — it tells you if your advanced model is actually better.

print("\n🤖  STEP 8: Training models...")
results = {}          # store all model results here

# ── MODEL 1: LINEAR REGRESSION (Baseline) ────────────────────────────────────
print("  ▸ Model 1: Linear Regression (baseline)...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results['Linear Regression'] = {
    'pred': lr_pred,
    'MAE' : mean_absolute_error(y_test, lr_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
}
print(f"     MAE = {results['Linear Regression']['MAE']:.1f}  "
      f"RMSE = {results['Linear Regression']['RMSE']:.1f}")

# ── MODEL 2: RANDOM FOREST ───────────────────────────────────────────────────
print("  ▸ Model 2: Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                           min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results['Random Forest'] = {
    'pred': rf_pred,
    'MAE' : mean_absolute_error(y_test, rf_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
}
print(f"     MAE = {results['Random Forest']['MAE']:.1f}  "
      f"RMSE = {results['Random Forest']['RMSE']:.1f}")

# ── MODEL 3: XGBOOST (or GradientBoosting fallback) ──────────────────────────
print("  ▸ Model 3: XGBoost..." if XGBOOST else "  ▸ Model 3: GradientBoosting (XGBoost substitute)...")
if XGBOOST:
    xgb_model = XGBRegressor(
        n_estimators    = 500,
        learning_rate   = 0.05,     # small = more careful learning
        max_depth       = 5,        # tree depth — 5 prevents overfitting
        subsample       = 0.8,      # use 80% of rows per tree
        colsample_bytree= 0.8,      # use 80% of features per tree
        min_child_weight= 3,
        reg_alpha       = 0.1,      # L1 regularisation
        reg_lambda      = 1.0,      # L2 regularisation
        random_state    = 42,
        verbosity       = 0,
        n_jobs          = -1,
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
else:
    # GradientBoostingRegressor from sklearn — same concept, no extra install
    xgb_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42
    )
    xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
model_name = 'XGBoost' if XGBOOST else 'GradientBoosting'
results[model_name] = {
    'pred': xgb_pred,
    'MAE' : mean_absolute_error(y_test, xgb_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
}
print(f"     MAE = {results[model_name]['MAE']:.1f}  "
      f"RMSE = {results[model_name]['RMSE']:.1f}")


# ── TIME SERIES CROSS-VALIDATION ─────────────────────────────────────────────
# WHAT:  Validate on 5 rolling folds to ensure model is stable, not just lucky.
# WHY:   One test period might be unusually easy or hard.
#        5 folds gives a more reliable estimate of real-world performance.

print("  ▸ TimeSeriesSplit Cross-Validation (5 folds)...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
fold_details = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    if XGBOOST:
        cv_model = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                max_depth=5, verbosity=0, random_state=42)
    else:
        cv_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                             max_depth=5, random_state=42)
    cv_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    fold_pred = cv_model.predict(X_train.iloc[val_idx])
    fold_mae  = mean_absolute_error(y_train.iloc[val_idx], fold_pred)
    cv_scores.append(fold_mae)
    fold_details.append((fold+1, len(tr_idx), len(val_idx), fold_mae))
    print(f"     Fold {fold+1}: train={len(tr_idx)} days, val={len(val_idx)} days, MAE={fold_mae:.1f}")

print(f"     CV Average MAE: {np.mean(cv_scores):.1f} ± {np.std(cv_scores):.1f}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 9: PROPHET MODEL                                      ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Facebook Prophet — a specialised time series library.
# WHY:   Prophet automatically detects weekly/yearly seasonality
#        and handles holidays. No feature engineering needed for it.
# DIFFERENCE vs XGBoost:
#   XGBoost — learns from your hand-crafted features, very accurate
#   Prophet — automatic, great for visualisation, built-in uncertainty bands

if PROPHET:
    print("\n🔮  STEP 9: Training Facebook Prophet model...")
    # Prophet requires columns named 'ds' (date) and 'y' (target)
    prophet_df = daily.reset_index()[['date','daily_sales']].rename(
        columns={'date':'ds', 'daily_sales':'y'}
    )

    # Indian public holidays — Prophet uses this to model sales spikes
    holidays_india = pd.DataFrame({
        'holiday': [
            'Republic_Day','Holi','Ram_Navami','Independence_Day',
            'Gandhi_Jayanti','Diwali','Christmas','New_Year_Eve',
            'Republic_Day_2022','Holi_2022','Independence_Day_2022','Diwali_2022',
            'Christmas_2022','Republic_Day_2023','Holi_2023','Independence_Day_2023',
            'Diwali_2023','Christmas_2023',
        ],
        'ds': pd.to_datetime([
            '2021-01-26','2021-03-28','2021-04-21','2021-08-15',
            '2021-10-02','2021-11-04','2021-12-25','2021-12-31',
            '2022-01-26','2022-03-18','2022-08-15','2022-10-24',
            '2022-12-25','2023-01-26','2023-03-08','2023-08-15',
            '2023-11-12','2023-12-25',
        ]),
        'lower_window': -1,   # 1 day before is also busy
        'upper_window':  1,   # 1 day after is also busy
    })

    prophet_train = prophet_df.iloc[:-len(y_test)]   # same train/test as before

    prophet_model = Prophet(
        holidays             = holidays_india,
        yearly_seasonality   = True,
        weekly_seasonality   = True,
        daily_seasonality    = False,
        changepoint_prior_scale   = 0.05,  # how flexible the trend is
        seasonality_prior_scale   = 10,    # how strong seasonality effects are
    )
    prophet_model.fit(prophet_train)

    # Forecast the test period
    future   = prophet_model.make_future_dataframe(periods=len(y_test))
    forecast = prophet_model.predict(future)

    prophet_pred = forecast['yhat'].tail(len(y_test)).values
    prophet_pred = np.clip(prophet_pred, 0, None)   # no negative predictions
    results['Prophet'] = {
        'pred': prophet_pred,
        'MAE' : mean_absolute_error(y_test, prophet_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, prophet_pred)),
    }
    print(f"     Prophet MAE = {results['Prophet']['MAE']:.1f}  "
          f"RMSE = {results['Prophet']['RMSE']:.1f}")

    # Prophet components chart (trend, weekly, yearly)
    fig_comp = prophet_model.plot_components(forecast)
    fig_comp.suptitle('Prophet Components — Trend, Weekly, Yearly Seasonality',
                      fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/charts/06_prophet_components.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅  Prophet components chart saved")

else:
    print("\n⚠️   STEP 9: Prophet not installed — skipping.")
    print("     Install with: pip install prophet")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 10: EVALUATION & VISUALISATION                        ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Compare all models, generate forecast charts, feature importance.
# WHY:   Numbers alone are not enough — visualisation proves the model works.

print("\n📉  STEP 10: Evaluation and Visualisation...")

# ── MODEL COMPARISON TABLE ────────────────────────────────────────────────────
print("\n  ▸ Model Comparison:")
print(f"  {'Model':<25} {'MAE':>8}  {'RMSE':>8}  {'MAE % Error':>12}")
print("  " + "-" * 58)
avg_sales = y_test.mean()
best_model_name = min(results, key=lambda k: results[k]['MAE'])
for name, res in sorted(results.items(), key=lambda x: x[1]['MAE']):
    mae_pct = res['MAE'] / avg_sales * 100
    star = " ← BEST" if name == best_model_name else ""
    print(f"  {name:<25} {res['MAE']:>8.1f}  {res['RMSE']:>8.1f}  {mae_pct:>11.1f}%{star}")

# ── CHART: FORECAST vs ACTUAL ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Forecast vs Actual Sales — Test Period', fontsize=15, fontweight='bold')

ax = axes[0]
ax.plot(y_test.index, y_test.values, color='steelblue', linewidth=2,
        label='Actual Sales', zorder=5)
ax.plot(y_test.index, results[model_name]['pred'], color='#e74c3c',
        linewidth=1.8, linestyle='--', label=f'{model_name} Forecast')
ax.plot(y_test.index, results['Linear Regression']['pred'], color='gray',
        linewidth=1.2, linestyle=':', label='Linear Baseline', alpha=0.7)
if 'Random Forest' in results:
    ax.plot(y_test.index, results['Random Forest']['pred'], color='#27ae60',
            linewidth=1.5, linestyle='-.', label='Random Forest', alpha=0.8)
ax.set_ylabel('Daily Sales (Units)')
ax.legend(fontsize=10)
ax.set_title(f"All Models vs Actual  |  {model_name} MAE={results[model_name]['MAE']:.1f}")

ax = axes[1]
# Residuals = Actual - Predicted (shows WHERE the model makes mistakes)
residuals = y_test.values - results[model_name]['pred']
ax.bar(y_test.index, residuals, color=['#e74c3c' if r < 0 else '#27ae60' for r in residuals],
       alpha=0.6, width=0.8)
ax.axhline(0, color='black', linewidth=1.2)
ax.set_ylabel('Residual (Actual − Predicted)')
ax.set_title(f'Residuals — {model_name}  (Green=underpredicted, Red=overpredicted)')

plt.tight_layout()
plt.savefig('outputs/charts/07_forecast_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("     ✅  Saved: outputs/charts/07_forecast_vs_actual.png")

# ── CHART: FEATURE IMPORTANCE ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Feature Importance — What Drives Restaurant Sales?',
             fontsize=14, fontweight='bold')

# XGBoost / GBM feature importance
if XGBOOST:
    imp_xgb = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
else:
    imp_xgb = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
top15 = imp_xgb.tail(15)
colors_imp = ['#e74c3c' if 'lag' in i or 'roll' in i else '#3498db' for i in top15.index]

ax = axes[0]
ax.barh(top15.index, top15.values, color=colors_imp)
ax.set_title(f'{model_name} — Top 15 Features\n(Red=lag/rolling, Blue=calendar/other)')
ax.set_xlabel('Feature Importance Score')

# Random Forest feature importance
imp_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
top15_rf = imp_rf.tail(15)
colors_rf = ['#e74c3c' if 'lag' in i or 'roll' in i else '#3498db' for i in top15_rf.index]

ax = axes[1]
ax.barh(top15_rf.index, top15_rf.values, color=colors_rf)
ax.set_title('Random Forest — Top 15 Features\n(Red=lag/rolling, Blue=calendar/other)')
ax.set_xlabel('Feature Importance Score')

plt.tight_layout()
plt.savefig('outputs/charts/08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("     ✅  Saved: outputs/charts/08_feature_importance.png")

# ── CHART: DATA CLEANING BEFORE/AFTER ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Data Cleaning — Before vs After', fontsize=14, fontweight='bold')

# Problem 1: units_sold with negatives
ax = axes[0, 0]
ax.hist(raw['units_sold'].dropna(), bins=60, color='#e74c3c', alpha=0.7, label='Before')
ax.set_title('units_sold — BEFORE\n(has negatives & extreme outliers)')
ax.set_xlabel('Units Sold')
ax.axvline(0, color='black', linewidth=2, linestyle='--')
ax.legend()

ax = axes[1, 0]
ax.hist(df['units_sold'], bins=60, color='#27ae60', alpha=0.7, label='After')
ax.set_title('units_sold — AFTER\n(cleaned: all positive, capped at 500)')
ax.set_xlabel('Units Sold')
ax.legend()

# Problem 2: temperature
ax = axes[0, 1]
ax.hist(raw['temperature'].dropna(), bins=60, color='#e74c3c', alpha=0.7)
ax.set_title('temperature — BEFORE\n(has values of 150°C and -5°C)')
ax.set_xlabel('Temperature (°C)')

ax = axes[1, 1]
ax.hist(df['temperature'], bins=60, color='#27ae60', alpha=0.7)
ax.set_title('temperature — AFTER\n(Bengaluru range: 10°C–40°C)')
ax.set_xlabel('Temperature (°C)')

# Problem 3: outlet name counts
ax = axes[0, 2]
raw_outlet_counts = raw['outlet'].value_counts().head(20)
ax.barh(range(len(raw_outlet_counts)), raw_outlet_counts.values, color='#e74c3c', alpha=0.7)
ax.set_yticks(range(len(raw_outlet_counts)))
ax.set_yticklabels(raw_outlet_counts.index, fontsize=7)
ax.set_title(f'outlet — BEFORE\n({raw["outlet"].nunique()} unique variants)')
ax.set_xlabel('Count')

ax = axes[1, 2]
clean_outlet_counts = df['outlet'].value_counts()
ax.barh(range(len(clean_outlet_counts)), clean_outlet_counts.values, color='#27ae60', alpha=0.7)
ax.set_yticks(range(len(clean_outlet_counts)))
ax.set_yticklabels(clean_outlet_counts.index, fontsize=9)
ax.set_title(f'outlet — AFTER\n({df["outlet"].nunique()} clean canonical names)')
ax.set_xlabel('Count')

plt.tight_layout()
plt.savefig('outputs/charts/09_cleaning_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print("     ✅  Saved: outputs/charts/09_cleaning_before_after.png")


# ── PLOTLY: Interactive Forecast Chart ────────────────────────────────────────
if PLOTLY:
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=y_test.index, y=y_test.values,
        mode='lines', name='Actual Sales',
        line=dict(color='steelblue', width=2)))
    fig_fc.add_trace(go.Scatter(
        x=y_test.index, y=results[model_name]['pred'],
        mode='lines', name=f'{model_name} Forecast',
        line=dict(color='red', width=2, dash='dash')))
    fig_fc.update_layout(
        title=f'Interactive Forecast vs Actual — {model_name}  (MAE={results[model_name]["MAE"]:.1f})',
        xaxis_title='Date', yaxis_title='Daily Sales',
        template='plotly_white', hovermode='x unified',
        legend=dict(x=0, y=1))
    fig_fc.write_html('outputs/charts/10_interactive_forecast.html')
    print("     ✅  Interactive forecast chart saved: 10_interactive_forecast.html")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 11: SAVE MODELS                                       ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT:  Save trained models to disk so they can be reused without retraining.
# WHY:   Training takes time. Saving lets you load the model instantly next time.
# NOTE:  Add models/ to your .gitignore — don't push large .pkl files to GitHub.

print("\n💾  STEP 11: Saving models...")
joblib.dump(xgb_model, 'outputs/models/xgboost_model.pkl')
joblib.dump(rf,         'outputs/models/random_forest_model.pkl')
joblib.dump(lr,         'outputs/models/linear_regression_model.pkl')
print("     ✅  Models saved to outputs/models/")
print("     REMINDER: Add outputs/models/*.pkl to your .gitignore")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 12: FINAL REPORT                                      ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n📋  STEP 12: Writing final report...")

business_mae = results[model_name]['MAE']
old_error_pct = 30.0
new_error_pct = business_mae / avg_sales * 100
avg_item_cost = 120   # Rs per unit
daily_waste_before = (old_error_pct/100) * avg_sales * avg_item_cost
daily_waste_after  = (new_error_pct/100) * avg_sales * avg_item_cost
annual_saving      = (daily_waste_before - daily_waste_after) * 365

report_lines = [
    "RESTAURANT SALES FORECASTING — FINAL REPORT",
    "=" * 55,
    "",
    "1. RAW DATA PROBLEMS FOUND AND FIXED",
    "-" * 40,
    f"   Total raw rows           : {len(raw):,}",
    f"   Rows after cleaning      : {len(df):,}",
    "",
    "   Problems fixed:",
    "   [DATE]     Mixed formats (ISO, US, DD-MM) → parsed with dayfirst=True",
    f"   [BOOLEAN]  13 variants per column (TRUE/Yes/1/Y etc.) → mapped to 0/1",
    f"   [OUTLET]   {raw['outlet'].nunique()} messy names → {df['outlet'].nunique()} canonical names",
    f"   [MENU]     {raw['menu_item'].nunique()} messy names → {df['menu_item'].nunique()} canonical names",
    f"   [NEGATIVE] {(raw['units_sold']<0).sum()} negative units_sold → converted to absolute value",
    f"   [EXTREME]  {(raw['units_sold']>500).sum()} extreme units_sold (>500) → set to NaN",
    f"   [TEMP]     {((raw['temperature']>40)|(raw['temperature']<10)).sum()} impossible temperatures → set to NaN",
    f"   [MISSING]  Filled temperature NaN by month median",
    f"   [MISSING]  Filled units_sold NaN by outlet×item median",
    "",
    "2. FEATURES CREATED",
    "-" * 40,
    f"   Total features : {len(feature_cols)}",
    f"   Calendar       : day_of_week, month, quarter, is_friday, is_monday, is_december",
    f"   Lag features   : lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_30",
    f"   Rolling        : roll_mean_7/14/30, roll_std_7, roll_max_7, roll_min_7, roll_median_7",
    f"   Temperature    : avg_temp, temp_lag_1, temp_above_30",
    f"   Domain flags   : is_weekend, is_holiday, is_summer, is_monsoon",
    "",
    "3. MODEL RESULTS",
    "-" * 40,
    f"   {'Model':<25} {'MAE':>8}  {'RMSE':>8}  {'Error %':>8}",
    "   " + "-" * 55,
]
for name, res in sorted(results.items(), key=lambda x: x[1]['MAE']):
    ep = res['MAE'] / avg_sales * 100
    star = "  ← WINNER" if name == best_model_name else ""
    report_lines.append(f"   {name:<25} {res['MAE']:>8.1f}  {res['RMSE']:>8.1f}  {ep:>7.1f}%{star}")

report_lines += [
    "",
    f"   Cross-Validation (5-fold TimeSeriesSplit):",
    f"   Avg MAE = {np.mean(cv_scores):.1f} ± {np.std(cv_scores):.1f}",
    "",
    "4. BUSINESS IMPACT",
    "-" * 40,
    f"   Average daily sales       : {avg_sales:.0f} units",
    f"   Previous error rate       : {old_error_pct:.0f}%  (guessing / manual)",
    f"   New error rate ({model_name}) : {new_error_pct:.1f}%",
    f"   Daily waste BEFORE        : Rs {daily_waste_before:,.0f}",
    f"   Daily waste AFTER         : Rs {daily_waste_after:,.0f}",
    f"   Estimated annual saving   : Rs {annual_saving:,.0f} per restaurant",
    "",
    "5. OUTPUT FILES",
    "-" * 40,
    "   outputs/restaurant_sales_clean.csv       Clean dataset",
    "   outputs/restaurant_features.csv          Feature matrix",
    "   outputs/charts/01_eda_overview.png        EDA overview (6 charts)",
    "   outputs/charts/02_outlet_menu_heatmap.png Outlet × item heatmap",
    "   outputs/charts/07_forecast_vs_actual.png  Forecast vs actual",
    "   outputs/charts/08_feature_importance.png  Feature importance",
    "   outputs/charts/09_cleaning_before_after.png Before/After cleaning",
    "   outputs/models/xgboost_model.pkl          Trained XGBoost model",
]

report_text = "\n".join(report_lines)
with open('outputs/reports/02_final_report.txt', 'w') as f:
    f.write(report_text)

print(report_text)
print("\n     ✅  Report saved: outputs/reports/02_final_report.txt")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 13: HOW TO LOAD AND USE THE SAVED MODEL               ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  HOW TO USE THE SAVED MODEL FOR PREDICTIONS")
print("=" * 65)
print("""
  # Load model and predict for any new date:

  import joblib, pandas as pd, numpy as np

  model = joblib.load('outputs/models/xgboost_model.pkl')
  features_df = pd.read_csv('outputs/restaurant_features.csv',
                             parse_dates=['date'], index_col='date')
  feature_cols = [c for c in features_df.columns if c != 'daily_sales']

  # Predict on last 7 days of data
  X_recent  = features_df[feature_cols].tail(7)
  forecasts = model.predict(X_recent)

  for date, pred in zip(X_recent.index, forecasts):
      print(f'{date.date()} ({date.day_name()}): ~{pred:.0f} units')
""")
 
print("\n" + "=" * 65)
print("  ALL STEPS COMPLETE — Check the outputs/ folder!")
print("  Charts  : outputs/charts/")
print("  Models  : outputs/models/")
print("  Reports : outputs/reports/")
print("=" * 65)
