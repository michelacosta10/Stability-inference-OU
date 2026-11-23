# Raw data

Expected layout (paths are set in `code/realdata/ou_core.py`):

```
PLANKTON_data/41467_2017_2571_MOESM6_ESM.xlsx                                  (included, public)
EEG_data/Sz1.mat                                                               (not included, public)
FINANCIAL_data/Combined_processed/Combined_processed/returns_wide_1s.csv       (not included, licensed)
```

- **Plankton:** Supplementary Data of Martin-Platero et al., Nat. Commun. 9, 266 (2018).
- **EEG:** SWEC-ETHZ iEEG database, short-term dataset, http://ieeg-swez.ethz.ch. We use the first patient (ID1),
  recording of the first seizure (`Sz1.mat`, 512 Hz, variable `EEG` with shape (T, N), N = 47 channels).
- **Financial:** 1-second mid-price log returns of US large-cap stocks, 2-13 March 2020 (index = timestamp,
  columns = ticker), built from LOBSTER and Databento order-book data. These data, and the results derived from
  them, are not included for licensing reasons; they can be obtained from LOBSTER and Databento.

The raw data are only needed by `bash run.sh realdata`.
