# US Nasdaq100 RD-Agent Configuration

## Purpose

This branch adapts the original CSI300 RD-Agent/Qlib benchmark to a US equity setting using Nasdaq100.

## Dataset

- Market: Nasdaq100
- Qlib provider_uri: `~/.qlib/qlib_data/us_data`
- Region: `us`
- Benchmark: `^NDX`
- Frequency: Daily
- Current available data range: 1999-12-31 to 2020-11-10

## Time Split

- Train: 2010-01-01 to 2017-12-31
- Validation: 2018-01-01 to 2019-12-31
- Test: 2020-01-01 to 2020-11-06

`2020-11-06` is used instead of `2020-11-10` to avoid Qlib calendar boundary errors at the final available trading date.

## Trading Cost Adjustment

The original CSI300 configuration used an A-share cost model:

- `open_cost: 0.0005`
- `close_cost: 0.0015`
- `min_cost: 5`
- `limit_threshold: 0.095`

For the US Nasdaq100 experiment, this was changed to:

- `open_cost: 0.0001`
- `close_cost: 0.0001`
- `min_cost: 0`
- `limit_threshold: null`

Rationale:

- US equities do not have the same daily price-limit rule as A-shares.
- US trading costs are not asymmetric in the same way as A-share buy/sell costs.
- The 1bp single-side cost is used as a simplified institutional transaction-cost assumption for the pilot experiment.

## Portfolio Construction

Current setting inherited from the original benchmark:

- `topk: 50`
- `n_drop: 5`

Important note:

For Nasdaq100, this may be too broad because 50 stocks represent around half of the universe. A more proportional setting may be:

- `topk: 20`
- `n_drop: 2`

This should be reviewed before the final US-market experiment.

## Files Modified

- `rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml`
- `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml`
- `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml`
- `rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml`
- `rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml`

## Experiment Status

A 1-loop Nasdaq100 RD-Factor test has been completed successfully.

Initial result:

- IC: 0.022938
- ICIR: 0.190675
- Rank IC: 0.020315
- Rank ICIR: 0.151905
- Annualized return with cost: -0.112678
- Max drawdown: -0.132189

The result did not replace the previous CSI300 SOTA, but it confirms that the Nasdaq100 RD-Factor pipeline is executable.
