# Impute & Validate

A hands-on walkthrough for handling missing values in a mixed numeric/categorical
dataset — and, more importantly, **measuring whether the imputation is actually
trustworthy** instead of just assuming it worked.

## Why this exists

Most imputation tutorials stop at "fill the blanks and move on." This repo goes
one step further: it validates imputation quality by artificially hiding known
values, imputing them, and measuring how close the guesses were to the truth —
compared against a naive baseline (mean/mode fill).

## Dataset

`synthetic_dataset.csv` — 100 rows, 12 columns, synthetically generated customer
transaction data (Kenyan context: M-Pesa, Card, Bank Transfer payment methods;
Nairobi, Mombasa, Kisumu, etc. as cities) with 5–20% missing values randomly
scattered per column, for practicing cleaning and imputation techniques.

| Column | Type | Notes |
|---|---|---|
| `customer_id` | string | unique ID, never imputed |
| `first_name`, `last_name` | string | filled with `"Unknown"` — mode-imputing names is meaningless |
| `age` | numeric | KNN imputed |
| `city` | categorical | mode imputed |
| `signup_date` | date | excluded from this pass |
| `product_category` | categorical | mode imputed |
| `purchase_amount_kes` | numeric | KNN imputed |
| `quantity` | numeric | KNN imputed |
| `payment_method` | categorical | mode imputed |
| `satisfaction_score` | numeric | KNN imputed |
| `is_repeat_customer` | boolean (string) | mode imputed |

## Approach

### 1. Numeric columns → `KNNImputer`
Fills missing numeric values by looking at the `k` nearest rows (by Euclidean
distance across other numeric columns) and averaging their values.

```python
from sklearn.impute import KNNImputer

knn_imp = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn_imp.fit_transform(df_num), columns=num_cols)
```

### 2. Categorical columns → `SimpleImputer`
Fills missing categorical values with the most frequent value in that column
(mode imputation). Name columns are deliberately excluded and filled with a
constant `"Unknown"` flag instead, since guessing a name from the mode produces
nonsensical duplicates.

```python
from sklearn.impute import SimpleImputer

cat_imp = SimpleImputer(strategy='most_frequent')
df_cat_filled = pd.DataFrame(cat_imp.fit_transform(df[mode_cols]), columns=mode_cols)

name_imp = SimpleImputer(strategy='constant', fill_value='Unknown')
df_names_filled = pd.DataFrame(name_imp.fit_transform(df[['first_name','last_name']]),
                                columns=['first_name','last_name'])
```

### 3. Validate — don't just trust it
Since you never know the "true" value of a real missing cell, validation works
by testing on cells where the true value **is** known:

1. Take complete rows (no missing values).
2. Artificially mask ~10% of known values.
3. Impute the masked version.
4. Compare imputed values against the real (hidden) ones.
5. Compute MAE (numeric) or accuracy (categorical).
6. Compare against a naive baseline (mean-fill / mode-fill).

```python
from sklearn.metrics import mean_absolute_error

complete_rows = df_num.dropna()
mask = np.random.rand(*complete_rows.shape) < 0.10
df_test = complete_rows.copy()
df_test[mask] = np.nan

df_test_imputed = pd.DataFrame(
    KNNImputer(n_neighbors=5).fit_transform(df_test),
    columns=complete_rows.columns
)

mae = mean_absolute_error(complete_rows.values[mask], df_test_imputed.values[mask])
```

**If your imputer doesn't beat the naive baseline, it's not adding value** —
use the simpler method instead.

## Key findings from this dataset

- KNN imputation on the numeric columns beat the mean-fill baseline overall
  (MAE ~653 vs. ~775), but per-column relative error was still high for
  `quantity` (~68% of its mean) and `age` (~50% of its mean) — meaning the
  filled values should be treated as reasonable placeholders, not ground truth.
- `satisfaction_score` had the lowest relative error (~34%), likely because its
  small discrete range (1–5) leaves less room for large misses.
- Mode imputation is fast and guarantees zero missing values, but it's a blunt
  tool — every missing `city`, for example, becomes the single most common
  city in the dataset with no per-row nuance.

## Requirements

```
pandas
numpy
scikit-learn
```

## Usage

```bash
pip install pandas numpy scikit-learn
python impute_and_validate.py
```

## License

MIT
