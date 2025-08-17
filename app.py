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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

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
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

app = Flask(__name__)

# Global variables
df = pd.DataFrame()
clf = None
data_version = 0  # Track data changes

# ---------------- Data Loading ----------------

def load_sales_data():
    global data_version
    if client is None or db is None or collection is None:
        return pd.DataFrame(columns=['Brand','Product_Name','Model','timestamp','Quantity_Sold','Unit_Price','Revenue','Year','Month','Product_ID','Rating'])
    
    records = list(collection.find({}))
    if not records:
        return pd.DataFrame(columns=['Brand','Product_Name','Model','timestamp','Quantity_Sold','Unit_Price','Revenue','Year','Month','Product_ID','Rating'])
    
    df = pd.DataFrame(records)
    data_version += 1  # Increment version when data changes

    # Ensure timestamp and numeric fields
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for col in ['Quantity_Sold','Unit_Price','Revenue','Year','Rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Load initial data
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
        'Unit_Price': 'mean',
        'Rating': 'mean'  # Added for ratings aggregation
    }).reset_index()
    monthly['Revenue'] = (monthly['Quantity_Sold'].fillna(0) * monthly['Unit_Price'].fillna(0)).astype(float)
    return monthly[['timestamp', 'Quantity_Sold', 'Revenue', 'Rating']]

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
        for m in range(1, 13):
            heat.append({'year': int(yr), 'month': int(m), 'value': int(row.get(m, 0))})
    return heat

# ------------- New Utilities for Ratings Graphs -------------

def average_rating_trend(df_sub):
    monthly = aggregate_monthly(df_sub)
    return {
        'labels': monthly['timestamp'].dt.strftime('%Y-%m').tolist(),
        'values': monthly['Rating'].fillna(0).astype(float).tolist()
    }

def rating_distribution(df_sub):
    ratings = df_sub['Rating'].dropna().astype(int)
    if len(ratings) == 0:
        return {'bins': [], 'values': []}
    hist, bins = np.histogram(ratings, bins=range(1, 7))  # Assuming ratings 1-5
    return {
        'bins': [f"{i}-{i+1}" for i in range(1, 6)],
        'values': hist.tolist()
    }

def ratings_vs_sales_scatter(df_sub):
    if 'Product_ID' not in df_sub.columns:
        return {'ratings': [], 'quantity_sold': [], 'revenue': []}
    
    grouped = df_sub.groupby('Product_ID').agg({
        'Rating': 'mean',
        'Quantity_Sold': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    return {
        'ratings': grouped['Rating'].fillna(0).tolist(),
        'quantity_sold': grouped['Quantity_Sold'].fillna(0).tolist(),
        'revenue': grouped['Revenue'].fillna(0).tolist()
    }

def top_rated_products(df_sub):
    grouped = df_sub.groupby(['Brand', 'Product_Name', 'Model']).agg({
        'Rating': 'mean',
        'Quantity_Sold': 'count'  # For weighting if needed
    }).reset_index()
    grouped = grouped.sort_values('Rating', ascending=False).head(10)
    labels = grouped.apply(lambda row: f"{row['Brand']} {row['Product_Name']} ({row['Model']})", axis=1).tolist()
    return {
        'labels': labels,
        'values': grouped['Rating'].fillna(0).tolist()
    }

def rating_heatmap_by_category(df_sub):
    tmp = df_sub.copy()
    tmp['Year'] = tmp['timestamp'].dt.year
    pivot = tmp.pivot_table(index='Brand', columns='Year', values='Rating', aggfunc='mean', fill_value=0)
    heat = []
    for brand, row in pivot.iterrows():
        for yr in pivot.columns:
            heat.append({'brand': brand, 'year': int(yr), 'value': float(row[yr])})
    return heat

def sentiment_breakdown(df_sub):
    ratings = df_sub['Rating'].dropna().astype(int)
    if len(ratings) == 0:
        return {'labels': ['Positive', 'Neutral', 'Negative'], 'values': [0, 0, 0]}
    
    positive = (ratings >= 4).sum()
    neutral = (ratings == 3).sum()
    negative = (ratings <= 2).sum()
    total = positive + neutral + negative
    return {
        'labels': ['Positive', 'Neutral', 'Negative'],
        'values': [positive / total * 100 if total else 0, 
                  neutral / total * 100 if total else 0, 
                  negative / total * 100 if total else 0]
    }

# ------------- ML Utilities for Buy/Not Buy -------------

def prepare_ml_data(df):
    """Simplified feature engineering with consistent features"""
    if df.empty:
        return None, None, None, None
    
    df_ml = df.dropna(subset=['Rating', 'Quantity_Sold']).copy()
    if len(df_ml) < 10:  # Need minimum samples
        return None, None, None, None
    
    # Target variable: Buy if rating >= 4
    df_ml['Buy'] = (df_ml['Rating'] >= 4).astype(int)
    
    # Use only basic features for consistency
    feature_cols = ['Rating', 'Quantity_Sold']
    
    X = df_ml[feature_cols]
    y = df_ml['Buy']
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_buy_classifier():
    """Train the buy/not buy classifier with basic features"""
    global clf, df
    
    if df.empty:
        logger.warning("No data to train ML model.")
        return False
    
    try:
        result = prepare_ml_data(df)
        if result[0] is None:  # Not enough data
            logger.warning("Not enough data to train ML model.")
            return False
        
        X_train, X_test, y_train, y_test = result
        
        if len(X_train) < 2:
            logger.warning("Insufficient training samples.")
            return False
        
        # Train RandomForest with basic parameters
        clf = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=8
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        logger.info(f"[ML] Buy Classifier trained with accuracy: {acc:.2f}")
        
        # Save model
        joblib.dump(clf, "buy_classifier.joblib")
        logger.info("[ML] Model saved to buy_classifier.joblib")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training ML model: {e}")
        return False

def load_or_train_classifier():
    """Load existing model or train new one"""
    global clf
    
    try:
        # Try loading existing model first
        clf = joblib.load("buy_classifier.joblib")
        logger.info("[ML] Loaded existing buy classifier")
        return True
    except FileNotFoundError:
        # Train new model if none exists
        success = train_buy_classifier()
        if success:
            logger.info("[ML] Trained new buy classifier")
        return success
    except Exception as e:
        logger.error(f"Error loading ML model: {e}")
        # Fallback to training new model
        return train_buy_classifier()

def get_buy_prediction(sub, user_budget=None, user_priority='balanced'):
    """Generate buy prediction with consistent features"""
    global clf, df
    
    if clf is None or sub.empty:
        return None, 'Model not available'
    
    try:
        # Calculate only basic features
        avg_rating = sub['Rating'].mean()
        total_qty = sub['Quantity_Sold'].sum()
        
        # Validate features
        if pd.isna(avg_rating) or pd.isna(total_qty):
            return None, 'Insufficient data for prediction'
        
        # Prepare features for prediction (matching training features)
        X_pred = pd.DataFrame({
            'Rating': [avg_rating],
            'Quantity_Sold': [total_qty]
        })
        
        # Get base prediction
        buy_prob = clf.predict_proba(X_pred)[0][1]
        
        # Apply personalization adjustments
        if user_priority == 'rating' and avg_rating >= 4.0:
            buy_prob = min(1.0, buy_prob + 0.1)  # Boost for high ratings
        elif user_priority == 'value' and user_budget and 'Unit_Price' in sub.columns:
            avg_price = sub['Unit_Price'].mean()
            if avg_price < user_budget:
                buy_prob = min(1.0, buy_prob + 0.15)  # Boost for good value
        elif user_priority == 'popularity' and total_qty > df['Quantity_Sold'].quantile(0.75):
            buy_prob = min(1.0, buy_prob + 0.1)  # Boost for popular items
        
        # Generate recommendation
        buy_rec = 'Buy' if buy_prob > 0.5 else 'Do not buy'
        
        # Add budget consideration
        if user_budget and 'Unit_Price' in sub.columns:
            avg_price = sub['Unit_Price'].mean()
            if avg_price > user_budget:
                buy_rec += f' (may exceed budget of ${user_budget:.2f})'
        
        return buy_prob, buy_rec
        
    except Exception as e:
        logger.error(f"Error in buy prediction: {e}")
        return None, 'Prediction failed'

# Initialize ML model
load_or_train_classifier()

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
    global df, data_version
    try:
        # Check if data needs reloading
        old_version = data_version
        df = load_sales_data()
        
        # Retrain model if data changed
        if data_version != old_version:
            load_or_train_classifier()

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

# ------------- Enhanced Route for Ratings Graphs -------------

@app.route('/api/ratings', methods=['POST'])
def ratings_api():
    global df, clf, data_version
    try:
        # Check if data needs reloading
        old_version = data_version
        df = load_sales_data()
        
        # Retrain model if data changed
        if data_version != old_version:
            load_or_train_classifier()

        data = request.get_json() or {}
        for k in ['brand', 'product', 'model', 'start_date', 'end_date']:
            if not data.get(k):
                return jsonify({'error': f'Missing field: {k}'}), 400

        # Get personalization inputs
        user_budget = float(data.get('budget', 0)) if data.get('budget') else None
        user_priority = data.get('priority', 'balanced')

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

        if sub.empty or sub['Rating'].dropna().empty:
            return jsonify({'error': 'No ratings data found for the selected criteria.'}), 400

        response = {
            'average_rating_trend': average_rating_trend(sub),
            'rating_distribution': rating_distribution(sub),
            'ratings_vs_sales_scatter': ratings_vs_sales_scatter(sub),
            'top_rated_products': top_rated_products(sub),
            'rating_heatmap': rating_heatmap_by_category(sub),
            'sentiment_breakdown': sentiment_breakdown(sub)
        }

        # Get buy prediction
        buy_prob, buy_rec = get_buy_prediction(sub, user_budget, user_priority)
        
        if buy_prob is not None:
            response['buy_probability'] = float(buy_prob)
            response['buy_recommendation'] = buy_rec
        else:
            response['buy_recommendation'] = buy_rec

        return jsonify(response)
    except Exception as e:
        logger.exception("ratings_api error")
        return jsonify({'error': str(e)}), 500

# ------------- New Route for Combined Recommendation -------------

@app.route('/api/recommendation', methods=['POST'])
def recommendation_api():
    global df, clf, data_version
    try:
        # Check if data needs reloading
        old_version = data_version
        df = load_sales_data()
        
        # Retrain model if data changed
        if data_version != old_version:
            load_or_train_classifier()

        data = request.get_json() or {}
        for k in ['brand', 'product', 'model', 'start_date', 'end_date']:
            if not data.get(k):
                return jsonify({'error': f'Missing field: {k}'}), 400

        # Get personalization inputs
        user_budget = float(data.get('budget', 0)) if data.get('budget') else None
        user_priority = data.get('priority', 'balanced')

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
            return jsonify({'error': 'No data found for the selected criteria.'}), 400

        # Forecast sales
        monthly = aggregate_monthly(sub).set_index('timestamp').asfreq('MS', fill_value=0).reset_index()
        steps = 6
        pred, chosen_model, conf, _ = select_best_model(monthly[['timestamp', 'Quantity_Sold']], steps=steps)

        # Get buy prediction
        buy_prob, buy_rec = get_buy_prediction(sub, user_budget, user_priority)

        response = {
            'forecast': {
                'labels': pd.date_range(monthly['timestamp'].iloc[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS').strftime('%Y-%m').tolist(),
                'values': [int(round(x)) for x in pred]
            },
            'buy_recommendation': buy_rec,
            'buy_probability': float(buy_prob) if buy_prob is not None else None,
            'forecast_model_used': chosen_model
        }
        return jsonify(response)

    except Exception as e:
        logger.exception("recommendation_api error")
        return jsonify({'error': str(e)}), 500

# ---------------- Health Check Route ----------------

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_model_loaded': clf is not None,
        'data_points': len(df),
        'brands_available': len(brands)
    })

# ---------------- Main ----------------

if __name__ == '__main__':
    app.run(debug=True)
