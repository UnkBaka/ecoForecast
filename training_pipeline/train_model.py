import os
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "models", "ecoForecast.db")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "weather_lstm_model.keras")

# NEW: We added weather_code to the inputs!
FEATURES = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'weather_code']
SEQ_LENGTH = 24


def map_wmo_to_class(code):
    """Simplifies complex weather codes into 4 main classes for the AI to learn."""
    if pd.isna(code): return 1  # Default to cloudy if missing
    code = int(code)
    if code == 0: return 0  # Clear
    if code in [1, 2, 3, 45, 48]: return 1  # Cloudy
    if 51 <= code <= 82: return 2  # Rain/Showers
    if code >= 95: return 3  # Thunderstorm
    return 1  # Default


def get_data():
    print("🔌 Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT {', '.join(FEATURES)} FROM weather_data ORDER BY timestamp ASC"
    df = pd.read_sql(query, conn)
    conn.close()

    # Map the complicated weather codes into our 4 simple AI classes
    df['weather_class'] = df['weather_code'].apply(map_wmo_to_class)

    print(f"📦 Loaded {len(df)} rows of training data.")
    return df


def create_sequences(data_scaled, weather_classes, seq_length):
    xs, y_temp, y_weather = [], [], []
    for i in range(len(data_scaled) - seq_length):
        x = data_scaled[i:(i + seq_length)]

        # Target 1: The next hour's temperature (Index 0)
        yt = data_scaled[i + seq_length][0]
        # Target 2: The next hour's weather class (0, 1, 2, or 3)
        yw = weather_classes[i + seq_length]

        xs.append(x)
        y_temp.append(yt)
        y_weather.append(yw)
    return np.array(xs), np.array(y_temp), np.array(y_weather)


def train():
    df = get_data()

    # Normalize all input features between 0 and 1
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[FEATURES].values)
    weather_classes = df['weather_class'].values

    # Create sequences
    X, y_temp, y_weather = create_sequences(data_scaled, weather_classes, SEQ_LENGTH)

    # Split into Train (80%) and Test (20%)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    yt_train, yt_test = y_temp[:train_size], y_temp[train_size:]
    yw_train, yw_test = y_weather[:train_size], y_weather[train_size:]

    print(f"🧠 Training Multi-Output Model on {len(X_train)} sequences...")

    # --- BUILD THE MULTI-OUTPUT MODEL ---
    # 1. The Shared "Brain" Layers
    inputs = Input(shape=(SEQ_LENGTH, len(FEATURES)))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    shared_dense = Dense(16, activation='relu')(x)

    # 2. Branch A: Predict Temperature (Regression -> 1 continuous number)
    temp_output = Dense(1, name='temp_out')(shared_dense)

    # 3. Branch B: Predict Weather Condition (Classification -> 4 categories)
    weather_output = Dense(4, activation='softmax', name='weather_out')(shared_dense)

    # Combine branches into one model
    model = Model(inputs=inputs, outputs=[temp_output, weather_output])

    # Compile with TWO different loss functions!
    model.compile(
        optimizer='adam',
        loss={
            'temp_out': 'mse',  # Mean Squared Error for Temp
            'weather_out': 'sparse_categorical_crossentropy'  # Crossentropy for Categories
        },
        metrics={
            'temp_out': 'mae',
            'weather_out': 'accuracy'
        }
    )

    # Train the dual model
    model.fit(
        X_train,
        {'temp_out': yt_train, 'weather_out': yw_train},
        validation_data=(X_test, {'temp_out': yt_test, 'weather_out': yw_test}),
        epochs=10,
        batch_size=32
    )

    # Save it
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Multi-Output Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()