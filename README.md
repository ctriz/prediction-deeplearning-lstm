# LSTM for Price Prediction
This repository implements a **sequence-to-one** LSTM model to predict the **next-day Close** price of a stock using daily **OHLCV** data enriched with **technical indicators** (RSI, MACD, Bollinger Bands). It includes a reproducible training pipeline, Bayesian hyperparameter tuning, evaluation/plotting, and single-step inference.

---

##  What’s Inside

- **Pure LSTM focus** (no mixed baselines) with a tidy Keras/TensorFlow pipeline.
- **Feature engineering** with `ta` library (RSI, MACD, Bollinger Bands).
- **Windowed sequences** (configurable lookback) → predict next-day `Close`.
- **Bayesian Optimization** for LSTM hyperparameters (`skopt`).
- **Saved artifacts** for reuse: `results/lstm_model.h5`, `results/scaler.pkl`, and tuning results.

---

## Repository structure

├─ data/  
│ └─ stock.csv # Daily OHLCV (Date, Symbol, Open, High, Low, Close, …)  
├─ results/  
│ ├─ bayes_opt_result.pkl # skopt result object (after tuning)  
│ ├─ lstm_model.h5 # trained Keras model  
│ └─ scaler.pkl # fitted MinMaxScaler for features  
├─ src/  
│ ├─ features.py # add_technical_indicators, select_features  
│ └─ model.py # build_lstm_model(...)  
├─ backtest.py # (optional) placeholder for strategy-level analysis  
├─ bayes_tuning.py # Bayesian hyperparameter search (skopt)  
├─ config.yaml # training & model hyperparameters  
├─ evaluate.py # loads model, plots Actual vs Predicted, RMSE  
├─ predict.py # single-step next-day Close prediction  
├─ requirements.txt  
└─ train.py

##  Data & Features

- Input: `data/stock.csv` (includes `Date`, OHLC, `Volume`, etc.).  
- Indicators computed in `src/features.py`:
  - `RSI`, `MACD` (diff), `BB_High`, `BB_Low`.
- Final model features (configurable):  
  `Open, High, Low, Close, Volume, RSI, MACD, BB_High, BB_Low`.

##  Configuration (`config.yaml`)

Key fields:

-   `features`: ordered list used for scaling + inverse transform.
    
-   `window_size`: lookback length (days).
    
-   `lstm_units`, `dropout_rate`, `dense_units`
    
-   `batch_size`, `epochs`, `validation_split`
    
-   `optimizer`, `loss`


##  Train

`python train.py` 

What it does:

1.  Loads `data/stock.csv`.
    
2.  Adds indicators → selects features → drops `NaN`.
    
3.  Scales with `MinMaxScaler` (fit on all configured features).
    
4.  Builds sequences of length `window_size` to predict next-day `Close`.
    
5.  Trains the LSTM with early stopping and saves:
    
    -   `results/lstm_model.h5`
        
    -   `results/scaler.pkl`

## Evaluate (RMSE + Plot)

`python evaluate.py` 

-   Loads model + scaler from `results/`.
    
-   Rebuilds sequences (fresh split).
    
-   Prints **Test RMSE** and shows **Actual vs Predicted Close** plot.

##  Predict Next-Day Close

`python predict.py` 

-   Uses the **latest `window_size` days** from `data/stock.csv`.
    
-   Prints: `Predicted Next Day Close Price: <value>`

##  Hyperparameter Tuning (Bayesian, with `skopt`)

`python bayes_tuning.py` 

-   Search space (narrow & practical):
    
    -   `lstm_units` ∈ [32, 64]
        
    -   `dropout_rate` ∈ [0.1, 0.3]
        
    -   `batch_size` ∈ [16, 32]
        
    -   `window_size` ∈ [20, 40]
        
-   Saves:
    
    -   Best config back into `config.yaml` (as plain numbers)
        
    -   Full result object → `results/bayes_opt_result.pkl`
        
> The objective runs short (few epochs, early stopping) for **speed**. For final training, re-run `train.py` with the tuned config.

##  Model Architecture

`Input: [window_size, num_features]
  → LSTM(lstm_units, return_sequences=True)
  → Dropout(dropout_rate)
  → LSTM(lstm_units)
  → Dropout(dropout_rate)
  → Dense(dense_units=1)     # next-day Close
Loss: mse   |   Optimizer: adam`

## Acknowledgements

-   `ta` technical analysis library
    
-   TensorFlow/Keras, scikit-learn, scikit-optimize
