## Problems

### How to deal with missing value
- I've found that treating 0's as missing and imputing values for them improved my LB score by about 100k over the non-imputed variables, irrespective of whether I treat them as numeric, categorical or ordinal categories.
I've used the mice package of R, set the inputs as only P1 - P37 (combined test and training for this purpose) and let it do the default imputation overnight (5 x 5), and used the median of the 5 outputs as the imputed value.
