So, I have removed the meta data features from features but when metadata sector grows would that be a problem for lgb model

Let's say I categories the stocks in 10-12 category
Having 10 to 12 static categories (like standard GICS sectors, e.g., Tech, Financials, Healthcare, Energy) is actually the ideal sweet spot for LightGBM. Because sector classifications are macro-level groups, they are bounded and won't endlessly expand like user IDs or text tags.
However, whether a growing or changing category list becomes a problem depends entirely on how you encode them before passing them into LightGBM.
The Two Ways to Encode Categories (And the Trap to Avoid)
1. The Danger: One-Hot Encoding (pd.get_dummies)
If you use one-hot encoding to turn your 10–12 sectors into 10–12 separate binary columns (is_tech, is_finance, etc.):
The Problem: If you add a new category later or if a test/inference batch encounters a stock from a sector that wasn't present during training, your feature matrix shape will change or mismatch. LightGBM will throw an error because the training columns and inference columns don't align.
2. The Safe Way: Native LightGBM Categorical or Integer Mapping
LightGBM has native support for categorical features, which handles this gracefully.
Instead of expanding columns, you map your 10-12 categories to integers (e.g., 0 through 11) and declare those columns as categorical_feature when calling .train() or fit().
Why this is safe: LightGBM treats them as discrete groups rather than continuous numbers, finding optimal splits across the groups without breaking your feature width if a category count remains stable.
Best Practices to Prevent Sector Issues
Keep the Mapping Dictionary Hardcoded & Static:
Define your 10–12 categories in a fixed configuration file or dictionary. Do not let your pipeline dynamically generate categories on the fly from incoming live data.
Python
SECTOR_MAPPING = {
    "Technology": 0,
    "Financials": 1,
    "Healthcare": 2,
    # ... up to 11/12
}
Handle Unseen Categories Gracefully:
If a new stock or obscure asset enters your universe that doesn't fit your 10-12 buckets, map it to an explicit "Other" or Unknown category (e.g., index 12) rather than letting it throw a KeyError or drop out.
Retrain Periodically:
Sectors rarely shift overnight, but companies can change primary lines of business (e.g., a company reclassifying from Industrials to Tech). Retraining your LightGBM model periodically (e.g., monthly or quarterly) ensures it adapts to any macro reclassifications.
As long as your 10–12 sectors are managed through a fixed integer mapping or LightGBM's native categorical type, scaling to 10-12 categories will cause zero problems and will actually give your LightGBM model a great macro-awareness boost!