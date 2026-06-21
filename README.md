NOMURA QUANT CHALLENGE  PROJECT OVERVIEW

This project presents a quantitative trading framework developed for the Nomura Quant Challenge.

The objective was to design, evaluate, and combine multiple trading strategies for a universe of 20 stocks while strictly avoiding look-ahead bias. The project consists of three major tasks:

1. Development of five independent trading strategies.
2. Construction of an ensemble strategy that dynamically selects the best-performing strategy.
3. Extension of the ensemble model to account for transaction costs and turnover.

All portfolio allocations satisfy the competition requirements:
- Total positive weights = +1
- Total negative weights = -1
- Future information is never used when generating weights.

DATASET
The challenge provides two datasets:

1. train_data.csv
   - First 3500 business days
   - Used for strategy development

2. cross_val_data.csv
   - Next 500 business days
   - Used for validation and testing

Each record contains:

- Date
- Symbol
- Open
- High
- Low
- Close

The stock universe consists of 20 stocks indexed from 0 to 19.

TASK 1: INDIVIDUAL STRATEGIES

Five independent alpha-generation strategies were implemented.

STRATEGY 1: WEEKLY RETURN MEAN REVERSION

OBJECTIVE : Identify stocks that have historically overperformed or underperformed over weekly horizons.

METHOD : 
1. Compute weekly returns over the previous 50 completed weeks.
2. Calculate average weekly return for every stock.
3. Rank all stocks by average return.

Portfolio Construction:

- Top 6 stocks receive short positions (-1/6 each).
- Bottom 6 stocks receive long positions (+1/6 each).
- Remaining stocks receive zero weight.

Hypothesis:

Historically strong performers may be overextended and likely to revert, while weak performers may recover.

STRATEGY 2: MOVING AVERAGE DIVERGENCE

Objective:

Compare short-term momentum against long-term trends.

Method:

LMA = Average closing price over previous 30 days

SMA = Average closing price over previous 5 days

Relative Position:

(SMA - LMA) / LMA

Portfolio Construction:

- Top 5 stocks by relative position are shorted.
- Bottom 5 stocks are bought.
- All others receive zero weight.

Hypothesis:

Short-term deviations from long-term trends tend to mean revert.

STRATEGY 3: RATE OF CHANGE (ROC)

Objective:

Capture short-term momentum extremes.

Method:

ROC = (Current Close - Close 7 Days Ago) / Close 7 Days Ago

Portfolio Construction:

- Top 4 ROC stocks receive short positions.
- Bottom 4 ROC stocks receive long positions.

Hypothesis:

Extreme momentum often reverses after reaching unsustainable levels.

STRATEGY 4: SUPPORT AND RESISTANCE

Objective:

Identify stocks trading near statistically significant support and resistance levels.

Method:

Support = SMA21 - 3 × Standard Deviation

Resistance = SMA21 + 3 × Standard Deviation

For every stock:

- Compute proximity to support.
- Compute proximity to resistance.

Portfolio Construction:

- 4 stocks closest to support are bought.
- 4 stocks closest to resistance are shorted.

Hypothesis:

Prices near support may rebound, while prices near resistance may decline.

STRATEGY 5: STOCHASTIC OSCILLATOR

Objective:

Measure price position relative to its recent trading range.

Method:

%K = 100 × (Close - Low14) / (High14 - Low14)

Portfolio Construction:

- Highest 3 %K stocks are shorted.
- Lowest 3 %K stocks are bought.

Hypothesis:

Overbought and oversold conditions frequently reverse.

BACKTESTING FRAMEWORK

A custom backtesting engine was implemented.

Features:

- Daily compounding
- Dynamic date handling
- Missing value protection
- Portfolio-level return aggregation
- Sharpe ratio calculation

Performance Metrics:

- Net Return (%)
- Sharpe Ratio

TASK 2: ADAPTIVE ENSEMBLE STRATEGY

Initial Approach:

A Random Forest classifier was initially tested to determine the best strategy.

However:
- Returns were weak.
- Sharpe ratio was negative.
- Generalization performance was poor.

The machine learning approach was therefore discarded.

FINAL APPROACH: ROLLING SHARPE SELECTOR

Motivation:

Different market conditions favor different strategies.

Method:
1. Calculate daily returns for each strategy.
2. Use a rolling 10-day lookback window.
3. Compute rolling Sharpe ratios.
4. Select the strategy with the highest Sharpe ratio.
5. Deploy that strategy for the next day.

Advantages:

- Adaptive to changing market conditions.
- Interpretable decision-making.
- No model training required.
- Lower risk of overfitting.

TASK 3: TRANSACTION COST AWARE ENSEMBLE

Real-world implementation requires accounting for trading costs.

TURNOVER CALCULATION

Turnover = Sum of absolute weight changes across all stocks.

Transaction Cost:

Cost = 1% × Turnover

IMPROVED ENSEMBLE LOGIC

A standard rolling Sharpe selector tends to switch strategies too frequently.

To reduce unnecessary trading activity, a sticky strategy selection mechanism was introduced.

The system remains with the current strategy unless:

Sharpe Improvement >
Switch Threshold + Turnover Penalty

Parameters:

Lookback Window = 10 Days
Switch Threshold = 0.25
Transaction Cost = 1%

Benefits:
- Lower turnover
- Reduced transaction costs
- More stable allocations
- Improved practical deployability

AVOIDING LOOK-AHEAD BIAS

The implementation strictly follows challenge requirements.

For every trading day:

- Only information available before the decision date is used.
- Future prices are never accessed.
- Ensemble selection uses only historical rolling performance.

This ensures realistic out-of-sample evaluation.

OUTPUT FILES

Task 1:
- task1.csv

Task 2:
- task2_weights.csv
- task_2.csv

Task 3:
- task3_weights.csv
- task_3.csv

RUNNING THE PROJECT

Execute:
python notebook.py

Current configuration runs:
task1()

To evaluate ensemble strategies, enable:

task2()
or
task3()

inside the main block.
KEY CONTRIBUTIONS

- Five independent quantitative trading signals.
- Dynamic regime-aware strategy selection.
- Rolling Sharpe ratio ensemble framework.
- Transaction-cost-aware optimization.
- Fully reproducible implementation.
- Strict prevention of look-ahead bias.
- Robust portfolio construction methodology.

CONCLUSION
This project evolved from a collection of standalone trading signals into a complete quantitative trading framework. By combining multiple alpha-generation methodologies with adaptive strategy selection and transaction-cost-aware optimization, the resulting system provides a realistic and systematic approach to portfolio management under changing market conditions.
