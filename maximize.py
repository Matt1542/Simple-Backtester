import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, ttest_1samp
from tqdm import tqdm

def load_config(config_path='config.json'):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def plot_eurirs_range(merged_df, lower_bound, upper_bound, config):
    # Extracting the relevant column names from the config
    date_column = config['column_mappings']['date']
    independent_column = config['column_mappings']['independent_parameter']
    dependent_column = config['column_mappings']['dependent_parameter']
    lower_bound = config['analysis_parameters']['lower_bound']
    upper_bound = config['analysis_parameters']['upper_bound']

    # Filter the data for the range between lower_bound and upper_bound for the dependent variable
    signal_range_df = merged_df[(merged_df[dependent_column] >= lower_bound) & (merged_df[dependent_column] <= upper_bound)]

    # Create a figure and a set of subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    # Plot the independent variable data on the first subplot
    ax1.plot(merged_df[date_column], merged_df[independent_column], label=independent_column, alpha=0.75)
    ax1.scatter(signal_range_df[date_column], signal_range_df[independent_column], color='red', label='Selected Range', alpha=1, zorder=5)
    ax1.set_title('Independent Variable with Highlighted Signals')
    ax1.set_ylabel(independent_column)
    ax1.legend()
    ax1.grid(True)

    # Plot the dependent variable data on the second subplot
    ax2.plot(merged_df[date_column], merged_df[dependent_column], label=dependent_column, alpha=0.75, color='green')
    ax2.fill_between(merged_df[date_column], lower_bound, upper_bound, color='green', alpha=0.3, label='Selected Range')
    ax2.set_title('Dependent Variable Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(dependent_column)
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap and show plots
    plt.tight_layout()
    plt.show()


def convert_units_to_numbers(value):
    """
    Convert values ending with 'K' or 'M' to their numerical equivalents.
    Additionally, handle values with commas as decimal separators.
    """
    try:
        if isinstance(value, str):
            value = value.replace(',', '.')
            if 'K' in value:
                return float(value.replace('K', '')) * 1000
            elif 'M' in value:
                return float(value.replace('M', '')) * 1000000
            return float(value)  # Convert strings to float if no 'K' or 'M'
        return value  # Return the value as is if it's not a string
    except ValueError:  # In case of conversion error, return NaN
        return np.nan

def clean_and_convert_percentages(df):
    """
    Converts percentage strings to float values in a DataFrame,
    handling both commas and periods as decimal separators.
    """
    for column in df.columns:
        if df[column].dtype == object and '%' in ''.join(df[column].astype(str)):
        
            df[column] = df[column].str.replace(',', '.').str.replace('%', '').astype(float)
    return df

def load_data(config):
    """
    Load and merge data from specified file paths in the configuration, dynamically handling column names.

    :param config: Configuration dictionary containing data paths and column mappings.
    :return: A merged DataFrame based on the specified configuration.
    """
    # Extract delimiters from the configuration
    variable_1_delimiter = config['delimiters']['variable_1_delimiter']
    independent_price_delimiter = config['delimiters']['independent_price_delimiter']
    
    # Load the datasets using the specified delimiters
    df1 = pd.read_csv(config['data_paths']['variable_1_path'], delimiter=variable_1_delimiter)
    df2 = pd.read_csv(config['data_paths']['independent_price_path'], delimiter=independent_price_delimiter)
    
    # Convert 'Date' columns to datetime objects without specifying the format
    df1[config['column_mappings']['date']] = pd.to_datetime(df1[config['column_mappings']['date']], errors='coerce')
    df2[config['column_mappings']['date']] = pd.to_datetime(df2[config['column_mappings']['date']], errors='coerce')
    
    # Clean and convert percentages, handling both commas and periods as decimal separators
    df1 = clean_and_convert_percentages(df1)
    df2 = clean_and_convert_percentages(df2)
    # Remove '%' from any column and convert to float, if applicable
    for column in df1.columns:
        if df1[column].dtype == object and df1[column].str.contains('%').any():
            df1[column] = df1[column].str.replace('%', '').astype(float)
    for column in df2.columns:
        if df2[column].dtype == object and df2[column].str.contains('%').any():
            df2[column] = df2[column].str.replace('%', '').astype(float)

    # Apply conversion for 'K' and 'M' values across all columns for both dataframes
    for column in df1.columns:
        df1[column] = df1[column].apply(convert_units_to_numbers)
    for column in df2.columns:
        df2[column] = df2[column].apply(convert_units_to_numbers)
    
    # Ensure all columns are of appropriate numeric type for analysis
    for column in df1.columns:
        if df1[column].dtype == object:
            df1[column] = pd.to_numeric(df1[column], errors='coerce')
    for column in df2.columns:
        if df2[column].dtype == object:
            df2[column] = pd.to_numeric(df2[column], errors='coerce')
    
    # Merge the two DataFrames on the 'Date' column
    merged_df = pd.merge(df1, df2, on=config['column_mappings']['date'], how='outer').sort_values(by=config['column_mappings']['date'],ascending=True)
    return merged_df

def perform_jarque_bera_test(returns, jb_p_value_threshold, plot=False ):
    jb_test_statistic, jb_p_value = jarque_bera(returns.dropna())  # Ensure to drop NaN values for accurate testing
    print(f"Jarque-Bera Test Statistic: {jb_test_statistic}, Jarque-Bera Test P-Value: {jb_p_value}")
    if jb_p_value < jb_p_value_threshold:
        print("The returns distribution is significantly different from a normal distribution.")
    else:
        print("No significant deviation from a normal distribution is found.")
    
    if plot:
        # Plotting the histogram
        plt.hist(returns.dropna(), bins=30, alpha=0.75, color='blue', edgecolor='black')
        plt.title('Returns Distribution')
        plt.xlabel('Return %')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

def calculate_sharpe_ratio(returns, period_days, config):
    risk_free_rate_annual = config['analysis_parameters']['risk_free_rate']
    risk_free_rate_period = (risk_free_rate_annual * period_days) / 365
    excess_returns = returns - risk_free_rate_period
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio


def calculate_sortino_ratio(returns, period_days, config):
    risk_free_rate_annual = config['analysis_parameters']['risk_free_rate']
    risk_free_rate_period = (risk_free_rate_annual * period_days) / 365
    downside_returns = returns[returns < risk_free_rate_period]
    downside_risk = downside_returns.std()
    expected_return = returns.mean()
    sortino_ratio = (expected_return - risk_free_rate_period) / downside_risk if downside_risk != 0 else float('inf')
    return sortino_ratio

def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min() if not returns.empty else float('nan')

def calculate_statistics(merged_df, config):
    periods = config['analysis_parameters']['periods']
    lower_bound = config['analysis_parameters']['lower_bound']
    upper_bound = config['analysis_parameters']['upper_bound']
    risk_free_rate = config['analysis_parameters']['risk_free_rate']  # Risk-free rate
    transaction_cost = config['analysis_parameters']['transaction_cost']  # Transaction cost

    dependent_parameter = config['column_mappings']['dependent_parameter']
    buy_signals = merged_df[(merged_df[dependent_parameter] >= lower_bound) & (merged_df[dependent_parameter] <= upper_bound)]

    returns_column = config['column_mappings']['returns_column']
    buy_signals_returns = merged_df.loc[buy_signals.index, returns_column]

    for period in periods:
        period_return_key = f'{period}_Day_Return'
        independent_parameter = config['column_mappings']['independent_parameter']
        merged_df[period_return_key] = merged_df[independent_parameter].pct_change(periods=period)
        
        period_returns = merged_df.loc[buy_signals.index, period_return_key].dropna()

        mean_anytime_return = merged_df[period_return_key].mean()

        # Calculate Win Ratio
        wins = period_returns[period_returns > 0].count()
        total_signals = period_returns.count()
        win_ratio = wins / total_signals if total_signals > 0 else 0

        # Custom t-statistic calculation
        mean_signals_return = period_returns.mean()
        std_tot = merged_df[period_return_key].std()
        std_signals = period_returns.std()
        n_sample = merged_df.shape[0]
        n_signals = period_returns.count()

        t_stat = (mean_signals_return - mean_anytime_return) / np.sqrt((std_tot**2 / n_sample) + (std_signals**2 / n_signals))

        # Use ttest_1samp for p-value
        test_result = ttest_1samp(period_returns, mean_anytime_return)
        p_value = test_result.pvalue  # Correctly access the pvalue attribute

        # Print the results
        print(f"Period {period}")
        print(f"Mean Signals Return({period}): {mean_signals_return:.2%}")
        print(f"Mean Anytime Return({period}): {mean_anytime_return:.2%}")
        print(f"Difference({period}): {mean_signals_return - mean_anytime_return:.2%}")
        print(f"Standard Deviation ({period}): {std_signals:.2%}")
        print(f"T-Stat({period}): {t_stat:.2f}, P-Value({period}): {p_value:.4f}")  # Corrected p-value formatting
        print(f"Sharpe Ratio({period}): {calculate_sharpe_ratio(period_returns, risk_free_rate):.2f}")
        print(f"Sortino Ratio({period}): {calculate_sortino_ratio(period_returns, risk_free_rate):.2f}")
        print(f"Maximum Drawdown({period}): {calculate_max_drawdown(period_returns):.2%}")
        print(f"Win Ratio({period}): {win_ratio:.2%}")
        print(f"Number of Samples (Signals): {total_signals}")
        print("\n" + "-"*80 + "\n")

def calculate_sharpe_ratio(returns, risk_free_rate):
    """
    Calculates the Sharpe Ratio for a series of returns.
    
    :param returns: A pandas Series of returns.
    :param risk_free_rate: Risk-free rate, expressed as a decimal.
    :return: The Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def calculate_sortino_ratio(returns, risk_free_rate):
    """
    Calculates the Sortino Ratio for a series of returns.
    
    :param returns: A pandas Series of returns.
    :param risk_free_rate: Risk-free rate, expressed as a decimal.
    :return: The Sortino Ratio.
    """
    downside_returns = returns[returns < risk_free_rate]
    expected_return = returns.mean()
    downside_risk = downside_returns.std()
    sortino_ratio = (expected_return - risk_free_rate) / downside_risk
    return sortino_ratio

def calculate_max_drawdown(returns):
    """
    Calculates the maximum drawdown for a series of returns.
    
    :param returns: A pandas Series of returns.
    :return: The maximum drawdown, expressed as a decimal.
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_net_returns(returns, transaction_cost):
    """
    Calculates net returns after accounting for transaction costs.
    
    :param returns: A pandas Series of returns.
    :param transaction_cost: Transaction cost, expressed as a decimal.
    :return: Net returns after transaction costs.
    """
    return returns - transaction_cost

def find_best_intervals(data_frame, config):
    interval_length = config['analysis_parameters']['interval_length']
    z_score_threshold = config['analysis_parameters']['z_score_threshold']
    periods = config['analysis_parameters']['periods']
    min_signals = config['analysis_parameters']['min_signals']
    independent_parameter = config['column_mappings']['independent_parameter']  # Dynamically get the independent parameter

    best_intervals = {}

    # Iterate over each column except the independent parameter and the date column
    for column in data_frame.columns:
        if column == independent_parameter or column == config['column_mappings']['date']:
            continue  # Skip the independent variable and date column

        print(f"\nAnalyzing {column}...")
        min_value = data_frame[column].min()
        max_value = data_frame[column].max()
        range_value = max_value - min_value
        dynamic_step = range_value / config['analysis_parameters']['iterations_per_step_calculation']

        for period in periods:
            best_mean_return = float('-inf')
            best_interval = None
            best_z_score = float('-inf')

            for start in np.arange(max_value, min_value - interval_length, -dynamic_step):
                end = start - interval_length

                buy_signal_indices = data_frame[(data_frame[column] >= end) & (data_frame[column] <= start)].index
                period_return_key = f'{period}_Day_Return'

                if period_return_key not in data_frame.columns:
                    continue

                buy_signal_returns = data_frame.loc[buy_signal_indices, period_return_key].dropna()
                if buy_signal_returns.empty or len(buy_signal_returns) < min_signals:
                    continue

                overall_mean = data_frame[period_return_key].mean()

                mean_return = buy_signal_returns.mean()
                std_dev = data_frame[period_return_key].std(ddof=1)
                standard_error = std_dev / np.sqrt(len(buy_signal_returns))
                z_score = (mean_return - overall_mean) / standard_error

                if z_score > z_score_threshold and mean_return > best_mean_return:
                    best_mean_return = mean_return
                    best_interval = (end, start)
                    best_z_score = z_score

            if best_interval:
                best_intervals[(column, period)] = {
                    'interval': best_interval,
                    'mean_return': best_mean_return,
                    'z_score': best_z_score
                }
                print(f"Best interval for {column} in period {period}: {best_interval}, Mean Return: {best_mean_return:.2%}, Z-score: {best_z_score:.2f}")
            else:
                print(f"No signals found at the given conditions for {column} in period {period}.")

    return best_intervals

def calculate_utility(return_rate, win_ratio, std, signal_count, total_data_points, config):
    alpha = config['analysis_parameters']['alpha']
    beta = config['analysis_parameters']['beta']
    gamma = config['analysis_parameters']['gamma']
    zeta = config['analysis_parameters']['zeta']
    utility = zeta * return_rate + alpha * win_ratio - beta * std + gamma * (signal_count / total_data_points)
    return utility

def find_optimal_signals_by_utility(data_frame, config):
    interval_length = config['analysis_parameters']['interval_length']
    z_score_threshold = config['analysis_parameters']['z_score_threshold']
    periods = config['analysis_parameters']['periods']
    min_signals = config['analysis_parameters']['min_signals']
    independent_parameter = config['column_mappings']['independent_parameter']
    total_data_points = data_frame.shape[0]
    
    # Calculate the total number of iterations for the overall progress bar
    analysis_columns = [col for col in data_frame.columns 
                    if independent_parameter not in col 
                    and 'day_return' not in col.lower() 
                    and col not in [config['column_mappings']['date'], 'Day_Return']]
    total_iterations = len(analysis_columns) * len(periods)
    
    optimal_signals = {}
    
    # Initialize the overall progress bar
    with tqdm(total=total_iterations, desc="Overall Progress", position=0) as pbar:
        for column in analysis_columns:
            print(f"\nAnalyzing {column}:")  
            
            for period in periods:
                best_signals = []
                period_return_key = f'{period}_Day_Return'
                
                if period_return_key not in data_frame.columns:
                    print(f"No period return key '{period_return_key}' found for column '{column}' and period {period}. Skipping...")
                    pbar.update(1)  # Update the overall progress bar
                    continue

                overall_mean = data_frame[period_return_key].mean()
                overall_std = data_frame[period_return_key].std()

                try:
                    min_value, max_value = data_frame[column].min(), data_frame[column].max()
                    num_steps = config['analysis_parameters']['iterations_per_step_calculation']
                    dynamic_step = (max_value - min_value) / num_steps

                    if dynamic_step <= 0:
                        raise ValueError("Dynamic step is non-positive. Check your interval_length and iterations_per_step_calculation settings.")
                    
                    signals_found = False  # Track if signals are found

                    # Iterate through possible start interval values

                    for start in tqdm(np.arange(min_value, max_value, dynamic_step), desc=f"Column '{column}', Period {period}", leave=False, position=1):
                        end = start + interval_length
                        signal_indices = data_frame[(data_frame[column] >= start) & (data_frame[column] < end)].index
                        signal_returns = data_frame.loc[signal_indices, period_return_key].dropna()

                        if len(signal_returns) >= min_signals:
                            signals_found = True  # Update the state to indicate signals are found
                            mean_return = signal_returns.mean()
                            win_ratio = (signal_returns > 0).sum() / len(signal_returns)
                            std = signal_returns.std()
                            signal_count = len(signal_returns)

                            utility = calculate_utility(mean_return, win_ratio, std, signal_count, total_data_points, config)

                            standard_error = overall_std / np.sqrt(len(signal_returns))
                            z_score = (mean_return - overall_mean) / standard_error

                            if z_score >= z_score_threshold and mean_return > std:
                                best_signals.append({'interval': (start, end), 'utility': utility, 'mean_return': mean_return,
                                                     'win_ratio': win_ratio, 'std_dev': std, 'z_score': z_score})

                    # End of interval analysis loop
                    if not best_signals:
                        if not signals_found:
                            print(f"\nNo signals found for column '{column}' and period {period}. Possible reasons: insufficient data points or values do not meet the analysis criteria.")
                        else:
                            print(f"\nNo positive signals found for column '{column}' and period {period} after applying thresholds. Potential reasons include unfavorable market conditions or overly stringent criteria.")
                    else:
                        # Sort signals by utility in descending order
                        best_signals.sort(key=lambda x: x['utility'], reverse=True)
                        # Limit output to the top 5 signals
                        print(f"\nTop 5 optimal signals for column '{column}' and period {period}:")
                        for signal in best_signals[:5]:  
                            print(f"Interval: {signal['interval']}, Utility: {signal['utility']}, Mean Return: {signal['mean_return']}, Win Ratio: {signal['win_ratio']}, Std Dev: {signal['std_dev']}, Z-Score: {signal['z_score']}")

                # Exception handling and progress bar update remains unchanged
                except Exception as e:
                    print(f"\nAn error occurred in the analysis for column '{column}' and period {period}: {e}")

                # Update the overall progress bar after completing each iteration
                pbar.update(1)


def main():
    config = load_config()  # Load configuration

    # Load and merge data based on configuration settings
    merged_df = load_data(config)

    # Dynamically calculate 'Next_Day_Return' using the independent parameter specified in config
    independent_parameter = config['column_mappings']['independent_parameter']
    
    merged_df[config['column_mappings']['returns_column']] = merged_df[config['column_mappings']['independent_parameter']].pct_change(periods=1)

    # Retrieve analysis bounds and periods from config
    lower_bound = config['analysis_parameters']['lower_bound']
    upper_bound = config['analysis_parameters']['upper_bound']
    periods = config['analysis_parameters']['periods']
    z_score_threshold = config['analysis_parameters']['z_score_threshold']
    risk_free_rate = config['analysis_parameters']['risk_free_rate']
    transaction_cost = config['analysis_parameters']['transaction_cost']
    # Plotting range data
    #print(merged_df.head)
    plot_eurirs_range(merged_df, lower_bound, upper_bound, config)

    # Calculating and printing statistics
    calculate_statistics(merged_df, config)

    perform_jarque_bera_test(merged_df[config['column_mappings']['returns_column']],
                             config['analysis_parameters']['jarque_bera_p_value_threshold'])

    #find_best_intervals(merged_df, config)
    find_optimal_signals_by_utility(merged_df, config)
if __name__ == "__main__":
    main()
