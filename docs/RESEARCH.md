# Research notes

## Dataset provenance and observed balance

This project uses the official UCI Machine Learning Repository **SMS Spam Collection**, cited as Almeida and Hidalgo (2011), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84). The file was downloaded from the [UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection), which lists a CC BY 4.0 license. The downloaded raw file contains 5,574 labeled English SMS messages: 4,827 ham and 747 spam. Exact duplicate labeled rows were removed before splitting (403 rows); the deduplicated dataset contains 5,171 rows (4,518 ham and 653 spam). No conflicting labels for an identical message were observed in this copy.

## Method choices

- All vectorizers are inside scikit-learn pipelines so vocabulary and IDF statistics are learned only from the fold's training messages. Scikit-learn's leakage guidance recommends fitting transformations on training subsets and using pipelines for cross-validation ([common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)).
- Model selection uses stratified five-fold cross-validation on the 70% training split. The validation split is reserved for choosing the decision threshold; the held-out test split is evaluated only after those choices.
- Spam precision is prioritized because false positives can incorrectly flag legitimate messages. The validation threshold maximizes spam recall while requiring spam precision of at least 0.98. Precision-recall curves summarize this threshold tradeoff ([scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)).
- Preprocessing is offline: Unicode NFKC normalization, regex tokenization, placeholders for URLs/contact details/currency/numbers, and scikit-learn's built-in English stop-word set.

## Known limitations and corpus issues

- The corpus is English-only and dates from the early 2010s. UCI describes it as a collection of SMS messages with ham/spam labels; it should not be assumed to represent modern traffic.
- Exact duplicate labeled rows are removed before splitting. The data loader also discards any identical message text carrying conflicting labels, to prevent text overlap across splits; none were found in the downloaded copy.
- The corpus reflects older UK and Singapore message sources. It may not transfer to current Indian OTP, KYC, UPI, delivery, or regional-language scams. This domain-shift concern is a limitation inferred from the corpus age and its published source context, not a measured benchmark result.
- The original class balance is about 13.4% spam. Deployment traffic may have a different prevalence, which changes precision even when ranking quality is unchanged.
- User-written modern spam examples in `docs/ANALYSIS.md` are **DEMO ONLY** and are excluded from all fitted models and reported metrics.

## References

1. Almeida, T. & Hidalgo, J. (2011). [SMS Spam Collection, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
2. Scikit-learn developers. [Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
3. Scikit-learn developers. [Classification metrics and precision-recall evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
4. Scikit-learn developers. [Model persistence security considerations](https://scikit-learn.org/stable/model_persistence.html).
5. Joblib maintainers. [`joblib.load` warning](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html).
