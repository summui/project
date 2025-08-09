import os
import warnings
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pymongo import MongoClient
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
load_dotenv()

# Optional: guarded import for auto_arima (can be disabled if not needed)
try:
    from pmdarima import auto_arima
    PMDARIMA_OK = True
except Exception:
    PMDARIMA_OK = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MongoDB Connection (safe for PyMongo 4) ---
MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("DB_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")

if not MONGODB_URI or not DB_NAME or not COLLECTION_NAME:
    raise RuntimeError("Missing MongoDB configuration. Set MONGODB_URI, DB_NAME, and COLLECTION_NAME in environment.")

client = MongoClient(MONGODB_URI)
# Do NOT use "if db" (truthy check) on PyMongo 4 objects
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

app = Flask(__name__)

# ---------------- Data Loading ----------------

def load_sales_data():
    # Defensive: ensure we have objects
    if client is None or db is None or collection is None:
        return pd.DataFrame(columns=['Brand','Product_Name','Model','timestamp','Quantity_Sold','Unit_Price','Revenue','Year','Month','Product_ID'])
    records = list(collection.find({}))
    if not records:
        return pd.DataFrame(columns=['Brand','Product_Name','Model','timestamp','Quantity_Sold','Unit_Price','Revenue','Year','Month','Product_ID'])
    df = pd.DataFrame(records)

    # Ensure timestamp and numeric fields
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for col in ['Quantity_Sold','Unit_Price','Revenue','Year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Month can be string; keep as-is
    return df

df = load_sales_data()

if not df.empty and 'timestamp' in df.columns:
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    min_date_str = min_ts.strftime('%Y-%m') if pd.notnull(min_ts) else None
    max_date_str = max_ts.strftime('%Y-%m') if pd.notnull(max_ts) else None
    brands = sorted(df['Brand'].dropna().astype(str).unique().tolist())
else:
    min_date_str = None
    max_date_str = None
    brands = []

# ------------- Utilities -------------

def aggregate_monthly(df_sub):
    # Aggregate monthly Quantity and Unit Price, compute Revenue proxy
    monthly = df_sub.resample('MS', on='timestamp').agg({
        'Quantity_Sold': 'sum',
        'Unit_Price': 'mean'
    }).reset_index()
    monthly['Revenue'] = (monthly['Quantity_Sold'].fillna(0) * monthly['Unit_Price'].fillna(0)).astype(float)
    return monthly[['timestamp', 'Quantity_Sold', 'Revenue']]

def build_seasonality(df_sub):
    tmp = df_sub.copy()
    tmp = tmp.dropna(subset=['timestamp'])
    if 'Quantity_Sold' not in tmp.columns or tmp.empty:
        return []
    tmp['Year'] = tmp['timestamp'].dt.year
    tmp['MonthNum'] = tmp['timestamp'].dt.month
    pivot = tmp.pivot_table(index='Year', columns='MonthNum', values='Quantity_Sold', aggfunc='sum', fill_value=0)
    heat = []
    for yr, row in pivot.iterrows():
        for m in range(1, 12+1):
            heat.append({'year': int(yr), 'month': int(m), 'value': int(row.get(m, 0))})
    return heat

# ------------- Forecast Models -------------

def holt_winters_forecast(monthly_sales, steps):
    y = monthly_sales['Quantity_Sold']
    if len(monthly_sales) >= 24 and y.sum() > 0:
        model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        name = "Holt-Winters Seasonal Model"
    else:
        model = ExponentialSmoothing(y, trend='add', seasonal=None)
        fit = model.fit()
        name = "Holt's Linear Trend Model"
    pred = pd.Series(fit.forecast(steps))
    pred[pred < 0] = 0
    upper = pred * 1.15
    lower = np.maximum(0, pred * 0.85)
    return pred.values, name, (upper.values, lower.values)

def arima_forecast(monthly_sales, steps):
    y = monthly_sales['Quantity_Sold']
    order = (1,1,1) if len(monthly_sales) < 3 else (2,1,2)
    fit = ARIMA(y, order=order).fit()
    pred = pd.Series(fit.forecast(steps))
    pred[pred < 0] = 0
    upper = pred * 1.15
    lower = np.maximum(0, pred * 0.85)
    return pred.values, "ARIMA", (upper.values, lower.values)

def auto_arima_forecast(monthly_sales, steps):
    if not PMDARIMA_OK:
        raise RuntimeError("pmdarima unavailable")
    y = monthly_sales['Quantity_Sold']
    model = auto_arima(y, seasonal=True, m=12, suppress_warnings=True, error_action="ignore")
    pred = model.predict(n_periods=steps)
    pred = np.where(pred < 0, 0, pred)
    upper = pred * 1.15
    lower = np.maximum(0, pred * 0.85)
    return pred, "Auto-ARIMA", (upper, lower)

def prophet_forecast(monthly_sales, steps):
    data = monthly_sales.rename(columns={'timestamp': 'ds', 'Quantity_Sold': 'y'})
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
    m.fit(data)
    future = m.make_future_dataframe(periods=steps, freq='MS')
    fr = m.predict(future)
    tail = fr.iloc[-steps:]
    yhat = tail['yhat'].clip(lower=0)
    upper = tail['yhat_upper'].clip(lower=0)
    lower = tail['yhat_lower'].clip(lower=0)
    return yhat.values, "Prophet", (upper.values, lower.values)

def backtest_rmse(series, model_func, horizon=3, folds=3):
    y = series['Quantity_Sold'].values
    if len(y) <= horizon + folds:
        return None
    rmses = []
    for i in range(folds):
        split_idx = len(series) - horizon - i
        if split_idx <= horizon or split_idx >= len(series):
            continue
        train = series.iloc[:split_idx].copy()
        test = series.iloc[split_idx: split_idx + horizon].copy()
        try:
            pred, _, _ = model_func(train, horizon)
            n = min(len(pred), len(test))
            rmse = mean_squared_error(test['Quantity_Sold'][:n], pred[:n]) ** 0.5
            rmses.append(rmse)
        except Exception:
            continue
    if not rmses:
        return None
    return float(np.mean(rmses))

def select_best_model(monthly_sales, steps=6):
    model_list = [prophet_forecast, holt_winters_forecast, arima_forecast]
    if PMDARIMA_OK:
        model_list.append(auto_arima_forecast)

    candidates = []
    for fn in model_list:
        try:
            bt_rmse = backtest_rmse(
                monthly_sales, fn,
                horizon=min(steps, 3),
                folds=min(3, max(1, len(monthly_sales)//6))
            )
            yhat, name, conf = fn(monthly_sales, steps)
            if bt_rmse is None and len(monthly_sales) > steps:
                train = monthly_sales.iloc[:-steps]
                test = monthly_sales.iloc[-steps:]
                try:
                    yhat_train, _, _ = fn(train, steps)
                    n = min(len(yhat_train), len(test))
                    bt_rmse = mean_squared_error(test['Quantity_Sold'][:n], yhat_train[:n]) ** 0.5
                except Exception:
                    bt_rmse = None
            candidates.append({'name': name, 'pred': yhat, 'conf': conf, 'rmse': bt_rmse})
        except Exception as e:
            logger.info(f"Model {fn.__name__} failed in selection: {e}")

    if not candidates:
        yhat, name, conf = holt_winters_forecast(monthly_sales, steps)
        comparison = [{'name': name, 'rmse': None, 'preview': [int(round(v)) for v in yhat]}]
        return yhat, name, conf, comparison

    candidates_sorted = sorted(candidates, key=lambda c: (float('inf') if c['rmse'] is None else c['rmse']))
    best = candidates_sorted[0]
    comparison = [{'name': c['name'], 'rmse': c['rmse'], 'preview': [int(round(v)) for v in c['pred']]} for c in candidates_sorted]
    return best['pred'], best['name'], best['conf'], comparison

# ---------------- Routes ----------------

@app.route('/')
def index():
    return render_template('index.html', brands=brands, min_date=min_date_str, max_date=max_date_str)

@app.route('/api/products/<brand>')
def get_products_for_brand(brand):
    if df.empty:
        return jsonify([])
    subset = df[df['Brand'].astype(str) == str(brand)]
    products = subset['Product_Name'].dropna().astype(str).unique().tolist()
    return jsonify(sorted(products))

@app.route('/api/models/<brand>/<product>')
def get_models_for_product(brand, product):
    if df.empty:
        return jsonify([])
    subset = df[
        (df['Brand'].astype(str) == str(brand)) &
        (df['Product_Name'].astype(str) == str(product))
    ]
    models = subset['Model'].dropna().astype(str).unique().tolist()
    return jsonify(sorted(models))

@app.route('/api/forecast', methods=['POST'])
def forecast_api():
    global df
    try:
        # Reload latest data each time
        df = load_sales_data()

        data = request.get_json() or {}
        for k in ['brand', 'product', 'model', 'start_date', 'end_date']:
            if not data.get(k):
                return jsonify({'error': f'Missing field: {k}'}), 400

        brand = str(data['brand'])
        product = str(data['product'])
        model_name = str(data['model'])
        start_date = pd.to_datetime(data['start_date'], errors='coerce')
        end_date = pd.to_datetime(data['end_date'], errors='coerce')
        if pd.isna(start_date) or pd.isna(end_date):
            return jsonify({'error': 'Invalid dates'}), 400

        sub = df[
            (df['Brand'].astype(str) == brand) &
            (df['Product_Name'].astype(str) == product) &
            (df['Model'].astype(str) == model_name) &
            (df['timestamp'] >= start_date) &
            (df['timestamp'] <= end_date)
        ].copy()

        if sub.empty:
            return jsonify({'error': 'No data found for the selected criteria. Please add data in your MongoDB cluster.'}), 400

        monthly = aggregate_monthly(sub)
        if len(monthly) < 2:
            return jsonify({'error': 'Not enough data for forecasting. Minimum 2 months required.'}), 400

        monthly = monthly.set_index('timestamp').asfreq('MS', fill_value=0).reset_index()

        steps = 6
        try:
            pred, chosen_model, conf, comparison = select_best_model(monthly[['timestamp', 'Quantity_Sold']], steps=steps)
        except Exception as e:
            logger.info(f"select_best_model failed: {e}")
            pred, chosen_model, conf = holt_winters_forecast(monthly[['timestamp', 'Quantity_Sold']], steps)
            comparison = [{'name': chosen_model, 'rmse': None, 'preview': [int(round(v)) for v in pred]}]

        last_date = monthly['timestamp'].iloc[-1]
        f_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

        # Revenue forecast proxy: mean Unit_Price of selected data
        last_price_series = sub['Unit_Price'].dropna().astype(float)
        price_proxy = float(last_price_series.mean()) if len(last_price_series) else 0.0
        f_revenue = (np.array(pred) * price_proxy).tolist()

        highs = np.round(conf[0]).astype(int).tolist()
        lows  = np.round(conf[1]).astype(int).tolist()
        vals  = np.round(pred).astype(int).tolist()

        response = {
            'history': {
                'labels': monthly['timestamp'].dt.strftime('%Y-%m').tolist(),
                'values': monthly['Quantity_Sold'].astype(int).tolist(),
                'revenue': [float(x) for x in monthly['Revenue'].astype(float).tolist()]
            },
            'forecast': {
                'labels': f_dates.strftime('%Y-%m').tolist(),
                'values': vals,
                'highs': highs,
                'lows': lows,
                'revenue': [float(x) for x in f_revenue]
            },
            'seasonality': build_seasonality(sub),
            'recommendation': {
                'message': f"{product} ({model_name}) by {brand} is projected to sell {int(np.round(pred).sum())} units in the next {steps} months. Model used: {chosen_model}",
                'model_used': chosen_model
            },
            'comparison': comparison
        }
        return jsonify(response)
    except Exception as e:
        logger.exception("forecast_api error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-comparison', methods=['POST'])
def model_comparison():
    try:
        data = request.get_json() or {}
        for k in ['brand', 'product', 'model', 'start_date', 'end_date']:
            if not data.get(k):
                return jsonify({'error': f'Missing field: {k}'}), 400

        brand = str(data['brand'])
        product = str(data['product'])
        model_name = str(data['model'])
        start_date = pd.to_datetime(data['start_date'], errors='coerce')
        end_date = pd.to_datetime(data['end_date'], errors='coerce')
        if pd.isna(start_date) or pd.isna(end_date):
            return jsonify({'error': 'Invalid dates'}), 400

        sub = df[
            (df['Brand'].astype(str) == brand) &
            (df['Product_Name'].astype(str) == product) &
            (df['Model'].astype(str) == model_name) &
            (df['timestamp'] >= start_date) &
            (df['timestamp'] <= end_date)
        ].copy()
        if sub.empty:
            return jsonify({'error': 'No data found for comparison'}), 400

        monthly = aggregate_monthly(sub)
        monthly = monthly.set_index('timestamp').asfreq('MS', fill_value=0).reset_index()
        if len(monthly) < 2:
            return jsonify({'error': 'Not enough data for comparison'}), 400

        models = [prophet_forecast, holt_winters_forecast, arima_forecast]
        if PMDARIMA_OK:
            models.append(auto_arima_forecast)

        out = []
        steps = 6
        for fn in models:
            try:
                rmse = backtest_rmse(monthly[['timestamp','Quantity_Sold']], fn, horizon=min(steps,3), folds=min(3, max(1, len(monthly)//6)))
                yhat, name, _ = fn(monthly[['timestamp','Quantity_Sold']], steps)
                out.append({
                    'name': name,
                    'rmse': None if rmse is None else float(rmse),
                    'forecast': [int(round(v)) for v in yhat]
                })
            except Exception as e:
                out.append({'name': fn.__name__, 'rmse': None, 'error': str(e), 'forecast': []})

        return jsonify({'models': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------- Main ----------------

if __name__ == '__main__':
    app.run(debug=True)
