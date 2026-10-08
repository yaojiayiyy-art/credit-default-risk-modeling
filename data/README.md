# Data access and schema

The source file is an instructor-simulated academic dataset. It is intentionally excluded from this public repository because no redistribution license was provided.

To reproduce the analysis, place the authorized course file at:

```text
data/raw/verizon_data.csv
```

Expected source fields:

| Field | Type | Meaning |
|---|---|---|
| `application_date` | date | Device-financing application date |
| `fico` | integer | Applicant FICO score |
| `device_cost` | numeric | Device price |
| `down_payment` | numeric | Initial payment at application |
| `payment_type` | category | Credit, Cash, or Bank transfer |
| `borrower_age` | integer | Applicant age in years |
| `del90` | category | 90+ delinquent/default or Performing/paid off |

The pipeline derives `dp_ratio`, `financed_amount`, and the binary modeling target. The code validates column presence, target labels, missing values, payment bounds, and the out-of-time split before training.
