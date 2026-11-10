# California Single-Family Home Price Prediction

Predicting the sale price of California single-family homes from 28 months of California Regional MLS (CRMLS) closed sales. Built during the IDX Exchange Data Science Internship.

The final model is XGBoost on 39 engineered features. Across five time-ordered validation folds it explains about 91% of the variance in log sale price, and its typical miss (median absolute percentage error) is 7.9%. A Streamlit app wraps the model so anyone can enter a home's size, beds, baths and lot size and get an estimate.

## Results at a glance

The baseline uses 15 raw MLS columns. Every model from the decision tree onward uses the engineered features described in [Feature engineering](#feature-engineering-04), so the jump from 0.34 to 0.86 reflects both the new features and the switch to a nonlinear model.

| Model | Features | Mean validation R² (log price) | Notes |
|---|---|---|---|
| Linear regression (baseline) | 15 raw | 0.34 | No feature engineering |
| Decision tree, `max_depth=10` | 39 engineered | 0.86 | Deeper trees overfit |
| Generalized additive model (pyGAM) | 34 engineered | 0.88 (test) | Overlapping columns removed; scored on April 2026 holdout, not CV |
| Random forest, `max_depth=30` | 39 engineered | 0.897 | |
| Random forest + XGBoost on its residuals | 39 engineered | 0.906 | No gain over XGBoost alone |
| **XGBoost**, `max_depth=8`, `lr=0.05`, 500 trees | 39 engineered | **0.907** | **Final model.** Best on all 5 folds |

Dollar-scale accuracy, 5 validation folds pooled:

| Model | MAPE | MdAPE (typical miss) |
|---|---|---|
| Random forest | 13.4% | 8.3% |
| XGBoost | 12.5% | 7.9% |

## Repository contents

| Notebook | What it does |
|---|---|
| `01_exploration.ipynb` | Loads and merges the 28 monthly CRMLS files, filters to single-family residential, explores the target and key features |
| `02_preprocessing.ipynb` | Removes impossible records, log-transforms skewed columns, imputes, drops leakage columns, builds the time-based train/test split and 5 CV folds |
| `03_baseline_model.ipynb` | Linear regression baseline |
| `04_model_comparison.ipynb` | Feature engineering (schools, comps, neighborhood prices, coast distance, and more), then decision tree vs random forest |
| `generalized_additive_model.ipynb` | GAM with partial-effect plots showing how each feature moves price |
| `05_advanced_models.ipynb` | XGBoost tuning and a random forest + XGBoost residual stack |
| `06_evaluation.ipynb` | MAPE, MdAPE and error by price band; tests a model scoped to $500k to $2.5M homes |
| `07_streamlit_app.ipynb` | Trains and saves the final model bundle, writes `app.py` and `requirements.txt` |

## Dataset

**Source.** 28 monthly CRMLS "Sold" exports (`CRMLSSold202401.csv` through `CRMLSSold202604.csv`), January 2024 to April 2026, provided through the IDX Exchange internship. Each row is one closed listing with about 80 columns (property details, location, listing and agent fields, dates and prices). Some months include `latfilled` / `lonfilled` flags for imputed coordinates.

**Size.**

| Stage | Rows |
|---|---|
| All 28 files merged | 615,707 |
| Residential, single-family only | 309,735 |
| After cleaning | 307,722 |
| After dropping non-California coordinates | 307,648 |

**Target.** `ClosePrice`, modeled as `log_ClosePrice`. Raw prices are heavily right-skewed (skew 2.77, median $893,500, mean $1.2M).

**External data used for feature engineering**

- [California Department of Education public school directory](https://www.cde.ca.gov/ds/si/ds/pubschls.asp) (`pubschls.txt`): school locations and levels
- [CAASPP 2025 research files](https://caaspp-elpac.ets.org/caaspp/ResearchFileListSB) (`sb_ca2025_all_csv_v1.txt`): share of students meeting or exceeding ELA and Math standards
- [Natural Earth 1:10m coastline](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/): downloaded automatically in `04`

## Preprocessing

**Cleaning (`02`).** 2,013 rows (0.65%) removed:

- Close price missing or zero
- Living area outside the 0.05th to 99.95th percentile
- Bedrooms or bathrooms outside 1 to 15
- Negative days on market
- Close price less than half or more than double the list price (likely typos)
- Price per square foot in the extreme 0.1% tails
- Duplicate listing keys, and duplicate address plus close date
- Year built before 1850 or after 2026 set to missing

**Transforms.** Log of `ClosePrice`; `log1p` of living area, lot size and building area.

**Missing values.** Yes/no amenity flags (view, waterfront, pool, fireplace, new construction) filled with `False`; missing HOA fee filled with 0; remaining numeric columns filled with the median and categorical with the mode. Eight fully empty columns dropped.

**Leakage removal.** List price, original list price, days on market, contract dates, agent and office fields, and system IDs were dropped. These exist only because a sale is underway, so the model should work for homes that aren't listed.

**Train/test split.** Chronological:

- **Test:** April 2026 closings (12,405 homes)
- **Train:** January 2024 to March 2026 (295,317 homes)
- **Validation:** 5 expanding-window folds inside the training period. Each fold trains only on sales that closed before it.

## Feature engineering (`04`)

I built these features in `04`, and every model from the decision tree onward is trained on them (39 columns in total, including the cleaned raw columns). The GAM uses 34 of them after dropping columns that duplicate each other.

| Feature | How it's built |
|---|---|
| `PropertyAge`, `log_PropertyAge` | 2026 minus year built |
| `ListSeason_sin`, `ListSeason_cos` | Listing date mapped onto a circle so December and January sit next to each other |
| `SchoolScore` | Average CAASPP rating of the nearest elementary, middle and high school, plus distance to each |
| `BedBathRatio`, `SqftPerBedroom`, `LotCoverage` | Layout and density ratios |
| `GeoCluster` | K-means (k = 20) on latitude and longitude |
| `HOA_Monthly`, `HasHOA` | HOA fee converted to a monthly amount |
| `Zip_ppsf_6m`, `Nbhd_ppsf_6m`, `Nbhd_logprice_est` | Average log price per sq ft in the same ZIP over the prior 6 months (falls back to `GeoCluster` under 5 sales) |
| `Comp_ppsf`, `Comp_logprice_est`, `Comp_size_diff`, `Comp_dist_km` | 10 nearest sales from the prior 12 months, similar to how appraisers pull comps |
| `CoastDist_km` | Distance to the nearest ocean coastline |

Neighborhood and comp features only look backward in time, so a home's own sale price never feeds its features. These replaced an earlier price-per-sq-ft feature that leaked the target.

## Models tested

The baseline runs on raw columns only. All other models use the engineered feature set.

**Linear regression (`03`).** Raw columns only, no engineered features. R² 0.34. Only 1.6% of predictions landed within $10,000 of the sale price, which set the bar for moving to nonlinear models.

**Decision tree and random forest (`04`).** Tuned `max_depth` across all 5 folds. The single tree peaked at depth 10 (R² 0.86) and got worse deeper. The forest (`max_samples=0.2`, `max_features=0.5`) kept improving to depth 30 (R² 0.897) because averaging many decorrelated trees cancels much of their variance.

**GAM (`generalized_additive_model.ipynb`).** A smooth curve per feature plus a latitude by longitude price surface, so each feature's effect can be read on its own. Test R² 0.878, typical miss 9.4%. What it shows:

- Location dominates: the location surface swings price by about 45% from one home to the next
- Waterfront adds about 13%, a private pool 6.5%, a fireplace 3.9%, a view 2.6%, all else equal
- Listing month moves price by at most about 1.8% over the year

It scores below the tree models because it doesn't learn interactions on its own (an extra 500 sq ft is worth more in Palo Alto than in Fresno).

**XGBoost (`05`).** Grid over `max_depth` (4, 6, 8), `learning_rate` (0.05, 0.1) and trees (250, 500, 1,000), with `subsample=0.8` and `colsample_bytree=0.8`. Best: depth 8, learning rate 0.05, 500 trees, mean R² 0.907. It beat the random forest on every fold and trains in seconds instead of minutes.

**Random forest + XGBoost stack (`05`).** XGBoost started from the forest's out-of-bag predictions to correct its errors. R² 0.906: no better than XGBoost alone, with two models to maintain.

## Evaluation (`06`)

**Error by price band (XGBoost)**

| Price band | Share of test | Test MdAPE | Validation MdAPE | Median signed error (test) |
|---|---|---|---|---|
| Under $500k | 14% | 7.1% | 8.3% | +4.2% |
| $500k to $750k | 22% | 5.6% | 6.2% | +0.1% |
| $750k to $1M | 21% | 6.2% | 7.0% | 0.0% |
| $1M to $1.5M | 20% | 7.9% | 8.6% | -1.5% |
| $1.5M to $2.5M | 15% | 8.9% | 9.3% | -2.8% |
| $2.5M and up | 8% | 10.5% | 12.8% | -7.3% |

The model is most accurate from $500k to $1M and pulls predictions toward the middle: it overpredicts cheap homes and underpredicts expensive ones.

**Scoping to $500k to $2.5M.** Retraining XGBoost only on homes in that range improved validation MdAPE from 7.5% to 7.0% and within-10% from 61.7% to 64.8%. This is an upper bound, since the range is set by actual sale price, which isn't known at prediction time.

## Known limitations

- **Test-set scores in `04`, `05`, `06` and `07` are optimistic.** Those notebooks refit the final model on all of `df_clean`, which still contains the April 2026 test homes, before scoring the test set. That's why test R² (0.93 for random forest, 0.946 for XGBoost) is higher than validation R². Use the validation numbers above when comparing models. The GAM notebook filters training to sales before April 2026, so its test score is a true holdout.
- **The app holds 35 of 39 features at their training median**, including location, schools and comps. Its estimate is for a typical California home with the entered size, so it can't distinguish a home in San Francisco from one in Bakersfield.
- **Intermediate CSVs aren't written by the notebooks in this repo.** See the file list below.

## How to re-run

The notebooks run in Google Colab and read from Google Drive.

**1. Set up the Drive folder.** Create this folder in your Drive and add the raw data:

```
MyDrive/IDX Exchange Data Science Internship/Google Colab Notebooks/
├── CRMLSSold202401_filled.csv ... CRMLSSold202604.csv   (28 monthly files)
├── pubschls.txt
└── sb_ca2025_all_csv_v1.txt
```

If you use a different location, update `BASE_PATH` (or `target_directory` in `01`) at the top of each notebook.

**2. Open each notebook in Colab** using the "Open in Colab" badge at the top, and allow Drive access when prompted. Colab already includes pandas, scikit-learn, XGBoost and matplotlib. `04` installs `geopandas` and the GAM notebook installs `pygam`. A GPU runtime speeds up `05` and `06` but isn't required.

**3. Run the notebooks in this order.** Each one reads files the previous step produced:

| Step | Notebook | Reads | Produces |
|---|---|---|---|
| 1 | `01_exploration` | 28 CRMLS CSVs | `filtered_residential_single_family_listings.csv` * |
| 2 | `02_preprocessing` | filtered listings | `X_for_pre_feat_engin.csv`, `validation_fold_1.csv` to `validation_fold_5.csv`, `test_dataset.csv` * |
| 3 | `03_baseline_model` | `validation_fold_1.csv`, `test_dataset.csv` | Baseline metrics |
| 4 | `04_model_comparison` | `X_for_pre_feat_engin.csv`, folds, test set, school files | `X_after_feat_engin.csv` **, `test_dataset_after_feat_eng.csv` |
| 5 | `generalized_additive_model` | Feature-engineered train and test CSVs | GAM results |
| 6 | `05_advanced_models` | Feature-engineered CSVs, folds | XGBoost results |
| 7 | `06_evaluation` | Same as `05` | Error tables and plots |
| 8 | `07_streamlit_app` | `X_after_feat_engin.csv` | `streamlit_app/` folder with the model and app |

\* The cells that save these files aren't in the committed notebooks. Add a `to_csv` call at the end of `01` and `02` before running end to end. In `01`, the `pd.read_csv` lines inside the loading loop are also commented out and need to be restored.

\** `04` saves `X_after_feat_engin.csv` to the Colab working directory. Change it to `f'{BASE_PATH}X_after_feat_engin.csv'` so later notebooks can find it.

Expect `04` to take about 20 minutes (comps and random forest tuning) and the GAM about 10.

## Launching the app

`07_streamlit_app.ipynb` writes three files to `streamlit_app/` in your Drive folder:

- `xgb_price_bundle.joblib`: the trained model, feature order, medians and input ranges
- `app.py`: the Streamlit app
- `requirements.txt`

**Run it locally**

1. Download the `streamlit_app/` folder from Drive.
2. Install dependencies. The XGBoost version is pinned to match the one that saved the model:
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. Open http://localhost:8501, enter living area, lot size, bedrooms and bathrooms, and click **Predict price**.

The app warns when an input falls outside the range of 98% of training homes (living area 740 to 5,566 sq ft, lot size 1,800 to 241,322 sq ft, 2 to 6 beds, 1 to 6 baths).

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, pyGAM, GeoPandas, matplotlib, seaborn, Streamlit, Google Colab.

## Author

Hana Kozuka · [GitHub](https://github.com/HKozuka) · [LinkedIn](https://linkedin.com/in/hana-kozuka)
