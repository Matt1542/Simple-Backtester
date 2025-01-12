Financial Analysis Utility Project
This Python project provides various statistical and financial analysis functionalities, with a focus on optimizing signal-based investment strategies using multiple performance metrics.

Main Functions:
Data Loading & Cleaning: The load_data function loads and merges financial datasets, handling CSV data with dynamic delimiters and cleaning methods. It also supports handling currency units like 'K' (thousands) and 'M' (millions) and percentage conversion.

Statistical Analysis:

Jarque-Bera Test: This test checks for deviations from normal distribution using perform_jarque_bera_test.
Sharpe and Sortino Ratios: Evaluates risk-adjusted returns and downside risk with calculate_sharpe_ratio and calculate_sortino_ratio.
Max Drawdown: Evaluates the risk of large portfolio losses over time using calculate_max_drawdown.
Signal & Interval Analysis:

Signal Intervals: Identifies optimal intervals using price signals and evaluates them by their Z-score with find_best_intervals.
Utility Calculation: Evaluates signals by a utility function that factors in return rates, win ratio, and volatility with calculate_utility.
Result Visualization: The plot_eurirs_range function helps visualize data ranges and signals, using time-series plots with highlighted intervals for both independent and dependent parameters.

Configurable Options:

Data Paths: Input CSV file locations.
Column Mappings: Date and parameters for analysis.
Analysis Parameters: Customizable thresholds, risk-free rates, and period lengths.
Requirements:
Libraries: json, pandas, numpy, matplotlib, scipy, tqdm.
This project can be used for analyzing financial datasets, optimizing trading strategies, and conducting performance assessments based on different risk metrics and time intervals.
