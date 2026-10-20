In prediction part, I provide it a day for prediction.

for example let's from 9:50AM to 11AM, now in that case it wont be able to match weights as that small data doesn't have enough data for 1h ema 9, 21 & 50 which we used in modal features.

so add such functionality that when required it fetches 30 days back data, and patch it chronologically, such that data I provide come at last in chronological order and it can calculate all ema and other indicators on all timeframe and perform the prediction with full confidence


When you pass a specific window (like 9:15 AM to 11:00 AM) to your prediction engine, the script cannot just look at those two hours in isolation. If it did, the 1H EMA 50 would be a blank NaN.
Instead, your prediction function handles this seamlessly behind the scenes using a Lookback Buffer. Here is the exact execution flow of how it works under the hood:
The 4-Step Inference Execution Flow
The Hidden Historical Fetch
Even though you specified 9:15 AM to 11:00 AM, your prediction script automatically tells your data loader: "Grab today's data, but also fetch the last 15–20 trading days of history leading up to today."
Global Indicator Calculation
The script runs your feature engineering pipeline across that entire combined dataset (the past weeks + today). Because the previous days are included, the 1-hour candles have already accumulated, meaning the 1H EMA 50 is fully calculated and active long before market open today.
The merge_asof Alignment
The higher-timeframe features are cleanly mapped onto your 1-minute rows across this entire timeline, ensuring zero lookahead bias.
The Target Slice & Prediction
Only after all the math is completely finished does your script slice the DataFrame to isolate your exact window: 9:15 AM to 11:00 AM.
Your GRU model takes the final sequence ending at 11:00 AM, reads all 252 features (including the fully valid 1H EMA 50), and outputs its Triple Barrier probability score.
Conceptual Code Blueprint
If you look at your live/inference script, it typically looks something like this:


```
def run_live_prediction(target_date):
  # 1. Fetch today + a safety buffer of past history (e.g., last 15 days)
  raw_df = fetch_data_with_lookback(target_date, days_back=15)

  # 2. Calculate multi-timeframe indicators globally (1m, 5m, 15m, 1h)
  features_df = engineer_features(raw_df)

  # 3. NOW slice the DataFrame to your exact intraday window
  inference_slice = features_df[
      (features_df["Timestamp"].dt.date == target_date)
      & (features_df["Timestamp"].dt.time >= datetime.time(9, 15))
      & (features_df["Timestamp"].dt.time <= datetime.time(11, 00))
  ]

  # 4. Feed into the GRU model safely!
  predictions = model.predict(inference_slice)
  return predictions
  ```

By pulling that brief historical buffer first, your 1H EMA 50 is fully warmed up and waiting the exact second your 9:15 AM candle loads!