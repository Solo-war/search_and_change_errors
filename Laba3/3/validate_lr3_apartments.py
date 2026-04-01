import pandas as pd

DATASET = "apartments_dataset_with_errors.csv"
ERRORS_MAP = "apartments_errors_map.csv"

df = pd.read_csv(DATASET)
err = pd.read_csv(ERRORS_MAP)

assert len(df) >= 300, "Должно быть минимум 300 записей"

features = [c for c in df.columns if c != "price_rub"]
assert 7 <= len(features) <= 10, "Должно быть 7–10 признаков"

error_rows = set(err["row_number_csv"].astype(int).tolist())
share = len(error_rows) / len(df)
print(f"Rows: {len(df)}; error rows: {len(error_rows)}; share: {share:.3%}")
assert 0.05 <= share <= 0.10, "Доля ошибок должна быть 5–10%"

print("OK")
