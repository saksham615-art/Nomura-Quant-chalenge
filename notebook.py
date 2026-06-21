#NOMURA QUANT CHALLENGE 2025

#The format for the weights dataframe for the backtester is attached with the question.
#Complete the below codes wherever applicable

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# def backtester_without_TC(weights_df):
#     #Update data file path here
#     data = pd.read_csv('cross_val_data.csv')

#     weights_df = weights_df.fillna(0)

#     start_date = 3500
#     end_date = 3999

#     initial_notional = 1

#     df_returns = pd.DataFrame(index=range(3500, 4000), columns=range(20), dtype=float)

#     for i in range(0, 20):
#         data_symbol = data[data['Symbol'] == i].sort_values('Date')
#         data_symbol = data_symbol.set_index('Date')
#         data_symbol = data_symbol['Close'].pct_change().fillna(0)
#         symbol_returns = data_symbol.loc[3500:3999]
#         df_returns[i] = symbol_returns
    
#     df_returns = df_returns.fillna(0)
    
#     weights_df = weights_df.loc[start_date:end_date]    
#     df_returns = df_returns.loc[start_date:end_date]

#     df_returns = weights_df.mul(df_returns)

#     notional = initial_notional

#     returns = []

#     for date in range(start_date,end_date+1):
#         returns.append(df_returns.loc[date].values.sum())
#         notional = notional * (1+returns[date-start_date])

#     net_return = ((notional - initial_notional)/initial_notional)*100
#     sharpe_ratio = (pd.DataFrame(returns).mean().values[0])/pd.DataFrame(returns).std().values[0]

#     return [net_return, sharpe_ratio]

#I changed the backtester a bit to dynamically build the data frame, else indexing issues would fill the df_returns with Nan values
def backtester_without_TC(weights_df, data_path='cross_val_data.csv'):
    data = pd.read_csv(data_path)

    weights_df = weights_df.fillna(0)
    test_dates = data[data['Symbol'] == 0]['Date'].values
    start_date = test_dates[0]
    end_date = test_dates[-1]

    initial_notional = 1

    df_returns = pd.DataFrame(index=test_dates, columns=range(20), dtype=float)
    for i in range(20):
        data_symbol = data[data['Symbol'] == i].sort_values('Date')
        data_symbol = data_symbol.set_index('Date')['Close']
        symbol_returns = data_symbol.pct_change().fillna(0)
        df_returns[i] = symbol_returns.loc[test_dates]

    df_returns = df_returns.fillna(0)

    weights_df = weights_df.loc[test_dates]
    df_returns = df_returns.loc[test_dates]

    df_returns = weights_df.mul(df_returns)

    notional = initial_notional
    returns = []

    for date in test_dates:
        daily_return = df_returns.loc[date].values.sum()
        returns.append(daily_return)
        notional = notional * (1 + daily_return)

    returns = np.array(returns)
    net_return = ((notional - initial_notional) / initial_notional) * 100
    if returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std()
    else:
        sharpe_ratio = 0

    return [net_return, sharpe_ratio]


def task1_Strategy1(data_path_train='train_data.csv', data_path_cv='cross_val_data.csv'):
    train_data = pd.read_csv(data_path_train)
    crossval_data = pd.read_csv(data_path_cv)

    # Use test period dates dynamically
    test_dates = crossval_data[crossval_data['Symbol'] == 0]['Date'].values
    output_df = pd.DataFrame(0.0, index=test_dates, columns=range(20))

    for week in range(0, (len(test_dates) + 4) // 5):
        week_start = week * 5
        week_end = min(week_start + 4, len(test_dates) - 1)
        if week_start > week_end: # No more weeks to process
            break

        mean_returns = []
        for symbol in range(20):
            #Get the close values
            train_close = train_data[train_data['Symbol'] == symbol].set_index('Date')['Close']
            crossval_close = crossval_data[crossval_data['Symbol'] == symbol].set_index('Date')['Close']
            full_close = pd.concat([train_close, crossval_close])
            prev_week_end_date = test_dates[week_start] - 1
            week_returns = []
            for w in range(50):
                this_week_end_date = prev_week_end_date - w * 5
                this_week_start_date = this_week_end_date - 4
                if this_week_start_date < 0: #
                    break
                last_close = full_close.get(this_week_end_date, np.nan)
                prev_last_close = full_close.get(this_week_start_date - 1, 1.0 if this_week_start_date - 1 < 0 else np.nan)
                if np.isnan(last_close) or np.isnan(prev_last_close) or prev_last_close == 0:
                    continue
                week_return = (last_close - prev_last_close) / prev_last_close
                week_returns.append(week_return)
            mean_return = np.mean(week_returns) if week_returns else 0
            mean_returns.append((symbol, mean_return))

        mean_returns.sort(key=lambda x: -x[1])
        top6 = [x[0] for x in mean_returns[:6]] #Dates corresponding to the best 6 stocks according to weekly returns
        bottom6 = [x[0] for x in mean_returns[-6:]] #Dates corresponding to the worst 6 stocks according to weekly returns

        for symbol in top6:
            output_df.iloc[week_start:week_end + 1, symbol] = -1/6
        for symbol in bottom6:
            output_df.iloc[week_start:week_end + 1, symbol] = 1/6

    return output_df


def task1_Strategy2(data_path_train='train_data.csv', data_path_cv='cross_val_data.csv'):
    train_data = pd.read_csv(data_path_train)
    crossval_data = pd.read_csv(data_path_cv)
    test_dates = crossval_data[crossval_data['Symbol'] == 0]['Date'].values
    output_df = pd.DataFrame(0.0, index=test_dates, columns=range(20))
    data = pd.concat([train_data, crossval_data], ignore_index=True)

    #Run loops for the dates according to given data
    for idx, date in enumerate(test_dates):
        relative_positions = []
        for symbol in range(20):
            symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
            #Now we get last 30 and 5 closes for LMA and SMA calculations
            closes_30 = symbol_data.loc[date - 30:date - 1] if (date - 30 >= 0) else symbol_data.loc[:date - 1]
            closes_5 = symbol_data.loc[date - 5:date - 1] if (date - 5 >= 0) else symbol_data.loc[:date - 1]
            if len(closes_30) < 30 or len(closes_5) < 5:
                relative_positions.append((symbol, 0))
                continue
            #Calculate LMA and SMA
            LMA = closes_30.mean()
            SMA = closes_5.mean()
            if LMA == 0 or np.isnan(LMA) or np.isnan(SMA):
                relative_positions.append((symbol, 0))
                continue
            #Calculate relative position
            relative_position = (SMA - LMA) / LMA
            relative_positions.append((symbol, relative_position))
        relative_positions.sort(key=lambda x: -x[1])
        top5 = [x[0] for x in relative_positions[:5]] #Dates corresponding to the best 5 stocks according to moving averages
        bottom5 = [x[0] for x in relative_positions[-5:]] # Dates corresponding to the worst 5 stocks according to moving averages
        for symbol in top5:
            output_df.iloc[idx, symbol] = -1/5
        for symbol in bottom5:
            output_df.iloc[idx, symbol] = 1/5
    return output_df


def task1_Strategy3(data_path_train='train_data.csv', data_path_cv='cross_val_data.csv'):
    train_data = pd.read_csv(data_path_train)
    crossval_data = pd.read_csv(data_path_cv)
    test_dates = crossval_data[crossval_data['Symbol'] == 0]['Date'].values
    output_df = pd.DataFrame(0.0, index=test_dates, columns=range(20))
    data = pd.concat([train_data, crossval_data], ignore_index=True)

    #Basic loop idea is same as for Strategy 2, only change is that now we calculate and sort by ROC instead
    for idx, date in enumerate(test_dates):
        rocs = []
        for symbol in range(20):
            symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
            #We check for missing or problematic data
            if (date - 7 < 0):
                rocs.append((symbol, 0))
                continue
            latest_close = symbol_data.get(date, np.nan)
            prev_close = symbol_data.get(date - 7, np.nan)
            if np.isnan(latest_close) or np.isnan(prev_close) or prev_close == 0:
                rocs.append((symbol, 0))
                continue
            roc = (latest_close - prev_close) / prev_close
            rocs.append((symbol, roc))
            #Sort rocs in descending order
        rocs.sort(key=lambda x: -x[1])
        top4 = [x[0] for x in rocs[:4]] #Dates corresponding to the best 4 stocks according to ROC
        bottom4 = [x[0] for x in rocs[-4:]] #Datescorresponding to the worst 4 stocks according to ROC
        for symbol in top4:
            output_df.iloc[idx, symbol] = -1/4 
        for symbol in bottom4:
            output_df.iloc[idx, symbol] = 1/4
    return output_df


def task1_Strategy4(data_path_train='train_data.csv', data_path_cv='cross_val_data.csv'):
    train_data = pd.read_csv(data_path_train)
    crossval_data = pd.read_csv(data_path_cv)
    test_dates = crossval_data[crossval_data['Symbol'] == 0]['Date'].values
    output_df = pd.DataFrame(0.0, index=test_dates, columns=range(20))
    data = pd.concat([train_data, crossval_data], ignore_index=True)

    for idx, date in enumerate(test_dates):
        proximities_support = []
        proximities_resistances = [] #These lists will help us calculate of latest closing prices with the support and resistance values
        for symbol in range(20):
            symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
            closes_21 = symbol_data.loc[date - 21:date - 1] if (date - 21 >= 0) else symbol_data.loc[:date - 1]
            if len(closes_21) < 21:
                proximities_support.append((symbol, np.inf))
                proximities_resistances.append((symbol, -np.inf))
                continue
            SMA = closes_21.mean()
            std = closes_21.std()
            support = SMA - 3 * std
            resistance = SMA + 3 * std
            latest_close = symbol_data.get(date, np.nan)
            if np.isnan(latest_close) or np.isnan(SMA) or np.isnan(std) or support == 0 or resistance == 0:
                proximities_support.append((symbol, np.inf)) #We assign an infinity value so as to ensure that this stock is not chosen
                proximities_resistances.append((symbol, -np.inf))
                continue
            prox_to_resistance = (latest_close - resistance) / resistance
            prox_to_support = (latest_close - support) / support
            proximities_support.append((symbol, prox_to_support))
            proximities_resistances.append((symbol, prox_to_resistance))
        #Ranking stocks based on proximity to support in increasing order
        proximity_support_sorted = sorted(proximities_support, key=lambda x: x[1])
        #We take the dates for the top 4 stocks according to proximity to support ranking
        top4_support = [x[0] for x in proximity_support_sorted[:4]]
        #Now, we will remove these from our data and then sort the remaining according to proximity to resistance in decreasing order
        support_vals = set(top4_support)
        remaining = [x for x in proximities_resistances if x[0] not in support_vals]
        remaining_sorted = sorted(remaining, key=lambda x: -x[1])
        #Finally, we take the dates for the top 4 stocks according to proximity to resistance ranking
        top4_resistance = [x[0] for x in remaining_sorted[:4]]
        #We assign weights of +1/4 to top 4 support stocks, and -1/4 to top 4 resistance stocks, and 0 to the rest
        for symbol in top4_support:
            output_df.iloc[idx, symbol] = 1/4 
        for symbol in top4_resistance:
            output_df.iloc[idx, symbol] = -1/4
    return output_df


def task1_Strategy5(data_path_train='train_data.csv', data_path_cv='cross_val_data.csv'):
    train_data = pd.read_csv(data_path_train)
    crossval_data = pd.read_csv(data_path_cv)
    test_dates = crossval_data[crossval_data['Symbol'] == 0]['Date'].values
    output_df = pd.DataFrame(0.0, index=test_dates, columns=range(20))
    data = pd.concat([train_data, crossval_data], ignore_index=True)

    for idx, date in enumerate(test_dates):
        k_metrics = []
        for symbol in range(20):
            symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
            #Last 14 closes
            closes_14 = symbol_data.loc[date - 14:date - 1] if (date - 14 >= 0) else symbol_data.loc[:date - 1]
            if len(closes_14) < 14:
                k_metrics.append((symbol, 0))
                continue
            #Now we find the highest and lowest stocks among these 14 days 
            low_14 = closes_14.min()
            high_14 = closes_14.max()
            close = symbol_data.get(date, np.nan)
            if np.isnan(close) or np.isnan(low_14) or np.isnan(high_14) or low_14 == high_14:
                k_metrics.append((symbol, 0))
                continue
            k = 100 * (close - low_14) / (high_14 - low_14)
            k_metrics.append((symbol, k))
        #There might be some cases where the close for the date is the same as the lowest for the 14, so we will remove any metrics where k = 0
        k_metrics = [x for x in k_metrics if not np.isnan(x[1])] #Since the 2nd value of each tuple is the k metric
        #Sort k metrics in decreasing order
        k_metrics.sort(key=lambda x: -x[1])
        #We get the dates corresponding to the top and bottom 3 k-metrics
        top3 = [x[0] for x in k_metrics[:3]] #Dates corresponding to the best 3 stocks according to k-metrics
        bottom3 = [x[0] for x in k_metrics[-3:]] # Dates corresponding to the worst 3 stocks according to k-metrics
        for symbol in top3:
            output_df.iloc[idx, symbol] = -1/3
        for symbol in bottom3:
            output_df.iloc[idx, symbol] = 1/3
    return output_df

def task1():
    Strategy1 = task1_Strategy1()
    Strategy2 = task1_Strategy2()
    Strategy3 = task1_Strategy3()
    Strategy4 = task1_Strategy4()
    Strategy5 = task1_Strategy5()

    performanceStrategy1 = backtester_without_TC(Strategy1)
    performanceStrategy2 = backtester_without_TC(Strategy2)
    performanceStrategy3 = backtester_without_TC(Strategy3)
    performanceStrategy4 = backtester_without_TC(Strategy4)
    performanceStrategy5 = backtester_without_TC(Strategy5)

    output_df = pd.DataFrame({'Strategy1':performanceStrategy1, 'Strategy2': performanceStrategy2, 'Strategy3': performanceStrategy3, 'Strategy4': performanceStrategy4, 'Strategy5': performanceStrategy5})
    output_df.to_csv('task1.csv')
    return


#Initially I used a random forest classifier to select the best strategy, but it returned bad results, with low returns and a negative sharpe
#ratio, so i changed my approach to a rolling sharpe selector, which selects the strategy with the highest sharpe ratio over a rolling window
def task2():
    output_df_weights = pd.DataFrame()
    
    Strategy1 = task1_Strategy1()
    Strategy2 = task1_Strategy2()
    Strategy3 = task1_Strategy3()
    Strategy4 = task1_Strategy4()
    Strategy5 = task1_Strategy5()
    strategies = [Strategy1, Strategy2, Strategy3, Strategy4, Strategy5]
    
    crossval_data = pd.read_csv('cross_val_data.csv')
    test_dates = crossval_data[crossval_data['Symbol'] == 0]['Date'].values
    #This value will prepare the window for which we apply the startegy, this can be changed as per requirement
    lookback = 20
    #Compute daily returns for each strategy
    all_returns = []
    data = pd.read_csv('cross_val_data.csv')
    for strat in strategies:
        daily_returns = []
        for date in test_dates:
            returns_today = []
            for symbol in range(20):
                symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
                try:
                    ret = (symbol_data.get(date, np.nan) - symbol_data.get(date - 1, np.nan)) / symbol_data.get(date - 1, np.nan)
                except:
                    ret = 0
                if np.isnan(ret) or np.isinf(ret):
                    ret = 0
                returns_today.append(ret)
            #Now we have the returns for all 20 stocks for that date, we will multiply them with the weights from the strategy
            weights = strat.loc[date].values if date in strat.index else np.zeros(20)
            r = np.sum(weights * returns_today)
            daily_returns.append(r)
        all_returns.append(np.array(daily_returns))
    all_returns = np.array(all_returns)
    
    # Rolling Sharpe Selector
    lookback = 10  # You can tune this window
    chosen_strategies = []
    for t in range(len(test_dates)):
        if t < lookback:
            chosen_strategies.append(0)  # Use Strategy 1 as default for first few days
        else:
            rolling_sharpes = []
            #We compute the rolling sharpe ratio of each startegy over the lookback period
            for s in range(5):
                window = all_returns[s, t-lookback:t]
                mean = window.mean()
                std = window.std() if window.std() > 0 else 1e-8  # Prevent division by zero
                sharpe = mean / std
                rolling_sharpes.append(sharpe)
            selected = np.argmax(rolling_sharpes)
            chosen_strategies.append(selected)

    # Create weights DataFrame for the selected strategies
    output_df_weights = pd.DataFrame(0.0, index=test_dates, columns=range(20))
    for t, date in enumerate(test_dates):
        strat_idx = chosen_strategies[t]
        output_df_weights.loc[date] = strategies[strat_idx].loc[date].values

    output_df_weights.to_csv('task2_weights.csv')
    results = backtester_without_TC(output_df_weights)
    df_performance = pd.DataFrame({'Net Returns': [results[0]], 'Sharpe Ratio': [results[1]]})
    df_performance.to_csv('task_2.csv')
    
    return



def calculate_turnover(weights_df):
    weights_diff_df = abs(weights_df-weights_df.shift(1))
    turnover_symbols = weights_diff_df.sum()
    turnover = turnover_symbols.sum()
    return turnover

# def backtester_with_TC(weights_df):
    #Update path for data here
    data = pd.read_csv('cross_val_data.csv')

    weights_df = weights_df.fillna(0)

    turnover = calculate_turnover(weights_df)

    start_date = 3000
    end_date = 3499

    transaction_cost = (turnover * 0.01)

    df_returns = pd.DataFrame()

    for i in range(0,20):
        data_symbol = data[data['Symbol']==i]
        data_symbol = data_symbol['Close']
        data_symbol = data_symbol.reset_index(drop=True)   
        data_symbol = data_symbol/data_symbol.shift(1) - 1
        df_returns =  pd.concat([df_returns,data_symbol], axis=1, ignore_index=True)
    
    df_returns = df_returns.fillna(0)
    
    weights_df = weights_df.loc[start_date:end_date]    
    df_returns = df_returns.loc[start_date:end_date]

    df_returns = weights_df.mul(df_returns)

    initial_notional = 1
    notional = initial_notional

    returns = []

    for date in range(start_date,end_date+1):
        returns.append(df_returns.loc[date].values.sum())
        notional = notional * (1+returns[date-start_date])

    net_return = ((notional - transaction_cost - initial_notional)/initial_notional)*100
    sharpe_ratio = (pd.DataFrame(returns).mean().values[0] - (transaction_cost/(end_date-start_date+1)))/pd.DataFrame(returns).std().values[0]

    return [net_return, sharpe_ratio]

def backtester_with_TC(weights_df, data_path='cross_val_data.csv'):
    data = pd.read_csv(data_path)

    weights_df = weights_df.fillna(0)
    test_dates = data[data['Symbol'] == 0]['Date'].values
    start_date = test_dates[0]
    end_date = test_dates[-1]

    initial_notional = 1

    df_returns = pd.DataFrame(index=test_dates, columns=range(20), dtype=float)
    for i in range(20):
        data_symbol = data[data['Symbol'] == i].sort_values('Date')
        data_symbol = data_symbol.set_index('Date')['Close']
        symbol_returns = data_symbol.pct_change().fillna(0)
        df_returns[i] = symbol_returns.loc[test_dates]

    df_returns = df_returns.fillna(0)

    # Align indices for weights and returns
    weights_df = weights_df.loc[test_dates]
    df_returns = df_returns.loc[test_dates]

    df_returns = weights_df.mul(df_returns)

    # Calculate turnover
    weights_diff_df = abs(weights_df - weights_df.shift(1).fillna(0))
    turnover = weights_diff_df.sum().sum()
    transaction_cost = (turnover * 0.01)

    notional = initial_notional
    returns = []

    for date in test_dates:
        daily_return = df_returns.loc[date].values.sum()
        returns.append(daily_return)
        notional = notional * (1 + daily_return)

    returns = np.array(returns)
    net_return = ((notional - transaction_cost - initial_notional) / initial_notional) * 100
    if returns.std() != 0:
        # Sharpe: subtract average daily TC from daily mean return
        sharpe_ratio = (returns.mean() - (transaction_cost / len(returns))) / returns.std()
    else:
        sharpe_ratio = 0

    return [net_return, sharpe_ratio]

# The task3 function implements an improved rolling Sharpe Selector with "sticky" logic and switching penalty
def task3():
    # Load strategies
    Strategy1 = task1_Strategy1()
    Strategy2 = task1_Strategy2()
    Strategy3 = task1_Strategy3()
    Strategy4 = task1_Strategy4()
    Strategy5 = task1_Strategy5()
    strategies = [Strategy1, Strategy2, Strategy3, Strategy4, Strategy5]

    crossval_data = pd.read_csv('cross_val_data.csv')
    test_dates = crossval_data[crossval_data['Symbol'] == 0]['Date'].values

    # Compute daily returns for each strategy
    all_returns = []
    data = pd.read_csv('cross_val_data.csv')
    for strat in strategies:
        daily_returns = []
        for date in test_dates:
            returns_today = []
            for symbol in range(20):
                symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
                try:
                    ret = (symbol_data.get(date, np.nan) - symbol_data.get(date - 1, np.nan)) / symbol_data.get(date - 1, np.nan)
                except:
                    ret = 0
                if np.isnan(ret) or np.isinf(ret):
                    ret = 0
                returns_today.append(ret)
            weights = strat.loc[date].values if date in strat.index else np.zeros(20)
            r = np.sum(weights * returns_today)
            daily_returns.append(r)
        all_returns.append(np.array(daily_returns))
    all_returns = np.array(all_returns)  # shape (5, len(test_dates))

    # Improved rolling Sharpe Selector with “sticky” logic and switching penalty
    lookback = 10
    chosen_strategies = []
    prev_weights = np.zeros(20)
    output_df_weights = pd.DataFrame(0.0, index=test_dates, columns=range(20))
    transaction_cost = 0.01  # 1% per unit turnover

    sharpe_switch_threshold = 0.25  # More conservative switching: increase if still too much turnover

    #For switching startegies, we only switch if the new strategy's Sharpe ratio is significantly better than the previous one
    for t in range(len(test_dates)):
        if t < lookback:
            chosen = 0
        else:
            rolling_sharpes = []
            for s in range(5):
                window = all_returns[s, t-lookback:t]
                mean = window.mean()
                std = window.std() if window.std() > 0 else 1e-8
                sharpe = mean / std
                rolling_sharpes.append(sharpe)
            best = np.argmax(rolling_sharpes)
            prev_idx = chosen_strategies[-1] if chosen_strategies else 0
            prev_sharpe = rolling_sharpes[prev_idx]
            best_sharpe = rolling_sharpes[best]
            turnover = np.sum(np.abs(strategies[best].loc[test_dates[t]].values - prev_weights))
            turnover_penalty = turnover * transaction_cost
            # Only switch if Sharpe improvement is much more than the penalty
            if (best != prev_idx) and ((best_sharpe - prev_sharpe) > (sharpe_switch_threshold + turnover_penalty)):
                chosen = best
            else:
                chosen = prev_idx
        weights_today = strategies[chosen].loc[test_dates[t]].values
        output_df_weights.loc[test_dates[t]] = weights_today
        prev_weights = weights_today.copy()
        chosen_strategies.append(chosen)

    output_df_weights.to_csv('task3_weights.csv')
    results = backtester_with_TC(output_df_weights)
    df_performance = pd.DataFrame({'Net Returns': [results[0]], 'Sharpe Ratio': [results[1]]})
    df_performance.to_csv('task_3.csv')
    return



if __name__ == '__main__':
    task1()
    # task2()
    # task3()