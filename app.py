from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
import os
from pymongo import MongoClient
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env
load_dotenv()
MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("DB_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def load_sales_data():
    records = list(collection.find({}))
    if not records:
        # Empty DataFrame with expected columns
        return pd.DataFrame(columns=['Brand', 'Product_Name', 'Model', 'timestamp', 'Quantity_Sold'])
    df = pd.DataFrame(records)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Initialize data at startup
df = load_sales_data()

# Derive min/max dates if data exists
if not df.empty:
    min_dt = df['timestamp'].min().strftime('%Y-%m')
    max_dt = df['timestamp'].max().strftime('%Y-%m')
    brands = sorted(df['Brand'].unique())
else:
    min_dt = None
    max_dt = None
    brands = []

# --- Utility: Model Selection & Forecasting ---

def holt_winters_forecast(monthly_sales, steps):
    if len(monthly_sales) >= 24:
        model = ExponentialSmoothing(
            monthly_sales['Quantity_Sold'],
            trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        model_repr = "Holt-Winters Seasonal Model"
    else:
        model = ExponentialSmoothing(monthly_sales['Quantity_Sold'],
                                     trend='add', seasonal=None)
        fit = model.fit()
        model_repr = "Holt's Linear Trend Model"
    pred = fit.forecast(steps)
    pred[pred < 0] = 0
    return pred, model_repr, fit

def arima_forecast(monthly_sales, steps):
    order = (1, 1, 1) if len(monthly_sales) < 3 else (2, 1, 2)
    model = ARIMA(monthly_sales['Quantity_Sold'], order=order)
    fit = model.fit()
    pred = fit.forecast(steps)
    pred[pred < 0] = 0
    return pred, "ARIMA", fit

def prophet_forecast(monthly_sales, steps):
    data = monthly_sales.rename(columns={'timestamp': 'ds', 'Quantity_Sold': 'y'})
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
    m.fit(data)
    future = m.make_future_dataframe(periods=steps, freq='MS')
    forecast = m.predict(future)
    yhat = forecast.iloc[-steps:]['yhat']
    yhat = yhat.apply(lambda x: max(0, x))
    upper = forecast.iloc[-steps:]['yhat_upper']
    lower = forecast.iloc[-steps:]['yhat_lower']
    return yhat, "Prophet", (upper, lower)

def select_best_model(monthly_sales, steps=6):
    y = monthly_sales['Quantity_Sold'].values
    train = monthly_sales.iloc[:-steps] if len(monthly_sales) > steps else monthly_sales
    test = monthly_sales.iloc[-steps:] if len(monthly_sales) > steps else monthly_sales
    candidates = []

    try:
        yhat, name, conf = prophet_forecast(train, steps)
        rmse = mean_squared_error(test['Quantity_Sold'][:len(yhat)], yhat[:len(test)]) ** 0.5
        candidates.append((rmse, name, yhat, conf))
    except Exception:
        pass

    try:
        yhat, name, fit = holt_winters_forecast(train, steps)
        rmse = mean_squared_error(test['Quantity_Sold'][:len(yhat)], yhat[:len(test)]) ** 0.5
        candidates.append((rmse, name, yhat, (yhat*1.15, yhat*0.85)))
    except Exception:
        pass

    try:
        yhat, name, fit = arima_forecast(train, steps)
        rmse = mean_squared_error(test['Quantity_Sold'][:len(yhat)], yhat[:len(test)]) ** 0.5
        candidates.append((rmse, name, yhat, (yhat*1.15, yhat*0.85)))
    except Exception:
        pass

    if not candidates:
        yhat, name, fit = holt_winters_forecast(monthly_sales, steps)
        return yhat, name, (yhat*1.15, yhat*0.85)

    candidates.sort(key=lambda x: x[0])
    yhat, chosen_name, conf = candidates[0][2], candidates[0][1], candidates[0][3]
    return yhat, chosen_name, conf

# --- Flask Endpoints ---

@app.route('/')
def index():
    return render_template(
        'index.html',
        brands=brands,
        min_date=min_dt,
        max_date=max_dt
    )

@app.route('/api/products/<brand>')
def get_products_for_brand(brand):
    if df.empty:
        return jsonify([])
    products = df[df['Brand'] == brand]['Product_Name'].unique().tolist()
    return jsonify(sorted(products))

@app.route('/api/models/<brand>/<product>')
def get_models_for_product(brand, product):
    if df.empty:
        return jsonify([])
    models = df[(df['Brand'] == brand) & (df['Product_Name'] == product)]['Model'].unique().tolist()
    return jsonify(sorted(models))

@app.route('/api/forecast', methods=['POST'])
def forecast_api():
    global df
    try:
        # Refresh data for latest DB changes
        df = load_sales_data()

        data = request.get_json()
        brand = data.get('brand')
        product = data.get('product')
        model = data.get('model')
        start_date = pd.to_datetime(data.get('start_date'))
        end_date = pd.to_datetime(data.get('end_date'))

        filtered_df = df[
            (df['Brand'] == brand) &
            (df['Product_Name'] == product) &
            (df['Model'] == model) &
            (df['timestamp'] >= start_date) &
            (df['timestamp'] <= end_date)
        ].copy()

        if filtered_df.empty:
            return jsonify({'error': 'No data found for the selected criteria. Please add data in your MongoDB cluster.'}), 400

        monthly_sales = filtered_df.resample('MS', on='timestamp')['Quantity_Sold'].sum().reset_index()
        monthly_sales.set_index('timestamp', inplace=True)
        monthly_sales = monthly_sales.asfreq('MS', fill_value=0)
        monthly_sales.reset_index(inplace=True)

        if len(monthly_sales) < 2:
            return jsonify({'error': 'Not enough data for forecasting. Minimum 2 months required.'}), 400

        forecast_steps = 6

        try:
            forecast, model_name, conf = select_best_model(monthly_sales, steps=forecast_steps)
        except Exception:
            forecast, model_name, conf = holt_winters_forecast(monthly_sales, forecast_steps)
            conf = (forecast * 1.15, forecast * 0.85)

        last_date = monthly_sales['timestamp'].iloc[-1]
        forecast_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=forecast_steps, freq='MS')

        highs = np.round(conf[0]).astype(int).tolist()
        lows = np.round(conf[1]).astype(int).tolist()

        response = {
            'history': {
                'labels': monthly_sales['timestamp'].dt.strftime('%Y-%m').tolist(),
                'values': monthly_sales['Quantity_Sold'].astype(int).tolist()
            },
            'forecast': {
                'labels': forecast_dates.strftime('%Y-%m').tolist(),
                'values': np.round(forecast).astype(int).tolist(),
                'highs': highs,
                'lows': lows
            },
            'recommendation': {
                'message': f"{product} ({model}) by {brand} is projected to sell {np.round(forecast).sum():.0f} units in the next 6 months. Model used: {model_name}",
                'model_used': model_name
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
