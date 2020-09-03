##this produces a new csv for all companies' price-data for release dates of MD&As

import math
from operator import concat
import numpy as np
import pandas as pd
import os
import pickle as pkl
import datetime as dt
from datetime import datetime as dtt


report_data_directory = 'D:/dis/report-pickle-data'
price_data_directory = 'D:/dis/price-csv-data'
output_folder = 'D:/dis/price-report-data-200000/'

list_of_tickers = pd.read_csv('D:/dis/company_report_price_metadata.csv', usecols=['Ticker:'])

def load_formatted_mda(ticker):
    price_filename = ticker + '.csv'
    report_filename = ticker + '.Pickle'
    print(report_filename)
    print(price_filename)
    # load pickle report data for current ticker
    infile = open(os.path.join(report_data_directory, report_filename), 'rb')
    loaded_report = pkl.load(infile)
    # load into a dataframe for the reports ticker and release data
    dataframe_report = loaded_report[['Ticker:', 'Released Date:']]
    dataframe_report['Released Time'] = pd.to_datetime(dataframe_report['Released Date:']).dt.strftime('%H:%M:%S')
    dataframe_report['Released Date:'] = pd.to_datetime(dataframe_report['Released Date:']).dt.strftime('%Y-%m-%d')
    dataframe_report['Released Date:'] = pd.to_datetime(dataframe_report['Released Date:'].astype(str),
                                                        format='%Y-%m-%d')

    # load price csv data for current ticker
    dataframe_price = pd.read_csv(os.path.join(price_data_directory, price_filename), usecols=['Date', 'Open', 'Close'])
    dataframe_price.rename({'Date': 'Released Date:'}, axis=1, inplace=True)
    # format price dataframe dates to match MD&A release dates
    dataframe_price['Released Date:'] = pd.to_datetime(dataframe_price['Released Date:'].astype(str), format='%Y%m%d')

    return dataframe_price, dataframe_report, price_filename, report_filename

def add_7_extra_dates(dataframe_report, report_7extraday_df):
    for index, row in dataframe_report.iterrows():
        # collect necessary data for new row entry to be added
        for date_increment in range(1, 370):
            ticker = row['Ticker:']
            released_date = row['Released Date:']
            released_date += dt.timedelta(days=date_increment)
            # set time of release to be a fixed time so that manually added date rows can be identified
            time = dt.time(23, 59, 59)
            # create new row in the form of a one row dataframe and add to end of the existing dataframe
            line = pd.DataFrame([[ticker, released_date, time]], columns=dataframe_report.columns)
            report_7extraday_df = report_7extraday_df.append(line)
    return report_7extraday_df

def add_370_extra_dates(dataframe_report,report_360pastday_df):
    for index, row in dataframe_report.iterrows():
        for date_increment in range(1, 370):
            ticker = row['Ticker:']
            released_date = row['Released Date:']
            released_date -= dt.timedelta(days=date_increment)
            # set time of release to be a fixed time so that manually added date rows can be identified
            time = dt.time(23, 59, 59)
            # create new row in the form of a one row dataframe and add to end of the existing dataframe
            line = pd.DataFrame([[ticker, released_date, time]], columns=dataframe_report.columns)
            report_360pastday_df = report_360pastday_df.append(line)
    return report_360pastday_df

def get_final_close_prices(dataframe_price_reports):
    for index, row in dataframe_price_reports.iterrows():
        release_time = row['Released Time']
        # check row is REAL MD&A release date
        if isinstance(release_time, str):
            # convert str timestamp to date-time type for time comparison
            formatted_timestamp = dtt.strptime(release_time, '%H:%M:%S').time()
            # check if the time truly a valid entry - not fake datetime entry
            if formatted_timestamp != dt.time(23, 59, 59):
                # If the time of report release is before 10AM we use Open and Close price on the day
                if formatted_timestamp < dt.time(9, 30, 00):
                    #Take the open and close price on the day
                    #Increment index (get the next open and close price, if release is on a non-trading day)
                    increment_index = 0
                    dataframe_price_reports.loc[index, 'Final open'] = dataframe_price_reports.loc[index, 'Open']
                    dataframe_price_reports.loc[index, 'Final close'] = dataframe_price_reports.loc[index, 'Close']
                    dataframe_price_reports.loc[index, 'Days after release until trading'] = 0
                    # if the current final prices have not yet been set, then continue to check next day price until valid price is set
                    while math.isnan(dataframe_price_reports.loc[index, 'Final open']):
                        increment_index += 1
                        dataframe_price_reports.loc[index, 'Final open'] = dataframe_price_reports.loc[index+increment_index, 'Open']
                        dataframe_price_reports.loc[index, 'Final close'] = dataframe_price_reports.loc[index+increment_index, 'Close']
                        dataframe_price_reports.loc[index, 'Days after release until trading'] = increment_index
                        if increment_index>29:
                            dataframe_price_reports.loc[index, 'Final open']=np.nan
                            dataframe_price_reports.loc[index, 'Final close'] =np.nan
                            break

                # If the time of report release is after 10AM we use Open and Close price on the next trading day
                else:
                    # get the row of the next day, and ensure it is a valid price value (isNaN check)
                    increment_index = 1
                    below_row = dataframe_price_reports.loc[index + increment_index]
                    dataframe_price_reports.loc[index, 'Final open'] = below_row['Open']
                    dataframe_price_reports.loc[index, 'Final close'] = below_row['Close']
                    dataframe_price_reports.loc[index, 'Days after release until trading'] = 1
                    # if the current final prices have not yet been set, then continue to check next day price until valid price is set
                    while math.isnan(dataframe_price_reports.loc[index, 'Final open']):
                        increment_index += 1
                        below_row = dataframe_price_reports.loc[index + increment_index]

                        dataframe_price_reports.loc[index, 'Final open'] = below_row['Open']
                        dataframe_price_reports.loc[index, 'Final close'] = below_row['Close']
                        dataframe_price_reports.loc[index, 'Days after release until trading'] = increment_index
                        if increment_index>29:
                            dataframe_price_reports.loc[index, 'Final open'] = np.nan
                            dataframe_price_reports.loc[index, 'Final close'] = np.nan
                            break
    return dataframe_price_reports

def get_x_day_momentum(dataframe_price_reports):
    for index, row in dataframe_price_reports.iterrows():
        release_time = row['Released Time']
        # momentum of stock price
        # 30 days
        # check there exists a value for the current MD&A 30 days prior to its release
        if isinstance(release_time, str) and (index >= 31):
            # set a decrement counter
            decrement = 30
            # set the 30 day momentum column (for the current row) equal to the close price 30 days ago (day of release).
            # while the value is NaN, continue to decrease the counter, (ie get the 29th day, then the 28th day... since MD&A release)
            while math.isnan(dataframe_price_reports.loc[index, '30 Day momentum']):
                # get the close price from the next soonest date.
                if not math.isnan(dataframe_price_reports.loc[index, 'Close']):
                    dataframe_price_reports.loc[index, '30 Day momentum'] = dataframe_price_reports.loc[
                                                                                index, 'Close'] - \
                                                                            dataframe_price_reports.loc[
                                                                                index - decrement, 'Close']
                    decrement -= 1
                    if (decrement < 20):
                        dataframe_price_reports.loc[index, '30 Day momentum'] = -90
                        break
                # Closing price on day of release may be NaN...
                else:
                    # print("no valid closing price found... using final close")
                    # print(dataframe_price_reports.loc[index, 'Released Date:'])
                    # print(dataframe_price_reports.loc[index - decrement, 'Close'])
                    dataframe_price_reports.loc[index, '30 Day momentum'] = dataframe_price_reports.loc[
                                                                                index, 'Final close'] - \
                                                                            dataframe_price_reports.loc[
                                                                                index - decrement, 'Close']
                    decrement -= 1
                    if (decrement < 20):
                        dataframe_price_reports.loc[index, '30 Day momentum'] = -90
                        # print(dataframe_price_reports.loc[index, '30 Day momentum'])
                        # print(dataframe_price_reports.loc[index, 'Released Date:'])
                        break

        # 180 days
        if isinstance(release_time, str) and (index > 181):
            decrement = 180
            while math.isnan(dataframe_price_reports.loc[index, '180 Day momentum']):
                if not math.isnan(dataframe_price_reports.loc[index, 'Close']):
                    dataframe_price_reports.loc[index, '180 Day momentum'] = dataframe_price_reports.loc[
                                                                                 index, 'Close'] - \
                                                                             dataframe_price_reports.loc[
                                                                                 index - decrement, 'Close']
                    decrement -= 1
                    if (decrement < 150):
                        dataframe_price_reports.loc[index, '180 Day momentum'] = -90
                        break
                else:
                    dataframe_price_reports.loc[index, '180 Day momentum'] = dataframe_price_reports.loc[
                                                                                 index, 'Final close'] - \
                                                                             dataframe_price_reports.loc[
                                                                                 index - decrement, 'Close']
                    decrement -= 1
                    if (decrement < 150):
                        dataframe_price_reports.loc[index, '180 Day momentum'] = -90
                        break
        # 360 days
        if isinstance(release_time, str) and (index > 361):
            decrement = 360

            while math.isnan(dataframe_price_reports.loc[index, '360 Day momentum']):
                if not math.isnan(dataframe_price_reports.loc[index, 'Close']):
                    dataframe_price_reports.loc[index, '360 Day momentum'] = dataframe_price_reports.loc[
                                                                                 index, 'Close'] - \
                                                                             dataframe_price_reports.loc[
                                                                                 index - decrement, 'Close']
                    decrement -= 1
                    print(decrement)
                    print(dataframe_price_reports.loc[index, 'Released Date:'])
                    dataframe_price_reports.loc[index - decrement, 'Released Date:']
                    if decrement < 330:
                        print(decrement)
                        print(dataframe_price_reports.loc[index, 'Released Date:'])
                        dataframe_price_reports.loc[index, '360 Day momentum'] = -90
                        break
                else:
                    dataframe_price_reports.loc[index, '360 Day momentum'] = dataframe_price_reports.loc[
                                                                                 index, 'Final close'] - \
                                                                             dataframe_price_reports.loc[
                                                                                 index - decrement, 'Close']
                    decrement -= 1
                    print(decrement)
                    dataframe_price_reports.loc[index - decrement, 'Released Date:']
                    if decrement < 330:
                        print(decrement)
                        print(dataframe_price_reports.loc[index, 'Released Date:'])
                        dataframe_price_reports.loc[index, '360 Day momentum'] = -90
                        break
    return dataframe_price_reports

def get_volatity(dataframe_price_reports):
    for index, row in dataframe_price_reports.iterrows():
        release_time = row['Released Time']
        # create column called volatility
        # perform std. dev. on price data (close) for 180 days prior to release
        if isinstance(release_time, str) and index > 181:
            # create empty list to store values from table slice
            last_180_days_price = []
            # get last 180 days of close price data in the list
            for x in range(1, 179):
                last_180_days_price.append(dataframe_price_reports.loc[index - x, ['Close']].values)

            # Use a new filtered list to remove index value and extract value from array
            last_180_days_price_good = []

            # Remove nan items and extract from list
            for item in last_180_days_price:
                last_180_days_price_good.append([item[0] for item in last_180_days_price if not np.isnan(item[0])])

            # calculate rounded std dev and set to current index value
            std_dev_180 = round(np.array(last_180_days_price_good[0]).std(), 3)
            dataframe_price_reports.loc[index, 'Volatility'] = std_dev_180
    return dataframe_price_reports

def get_returns(dataframe_price_reports):
    # add columnm for returns
    dataframe_price_reports = dataframe_price_reports.reindex(
        columns=dataframe_price_reports.columns.tolist() + ['Return'])
    for index, row in dataframe_price_reports.iterrows():
        # calculate return value and assign to 'Return' column for each remaining row.
        return_value = ((row['Final close'] - row['Final open']) / row['Final open']) * 100
        # round return value to 3 dp
        return_value = round(return_value, 3)
        #
        dataframe_price_reports.loc[index, 'Return'] = return_value
        # if dataframe_price_reports.loc[index, '30 Day momentum']==-1 or dataframe_price_reports.loc[index, '180 Day momentum']==-1 or dataframe_price_reports.loc[index, '360 Day momentum']==-1:
        #     dataframe_price_reports.drop(index, axis=1)
    return dataframe_price_reports

def produce_csv_all_tickers(list_of_tickers):
    for ticker in list_of_tickers['Ticker:'].values:
        dataframe_price, dataframe_report, price_filename, report_filename = load_formatted_mda(ticker)
        report_7extraday_df = pd.DataFrame(columns=dataframe_report.columns)
        report_360pastday_df = pd.DataFrame(columns=dataframe_report.columns)

        report_7extraday_df = add_7_extra_dates(dataframe_report,report_7extraday_df)
        report_360pastday_df = add_370_extra_dates(dataframe_report, report_360pastday_df)

        # Add together the original MD&A release date dataframe and extra_day values dataframe, then sort by date.
        dataframe_extra_days = report_7extraday_df.append(report_360pastday_df)

        dataframe_report_sorted_dates = dataframe_report.append(dataframe_extra_days)
        dataframe_report_sorted_dates = dataframe_report_sorted_dates.sort_values(by=['Released Date:'])
        dataframe_report_sorted_dates = dataframe_report_sorted_dates.drop_duplicates()

        # create a merged dataframe that adds price data to all selected dates (date of md&a release + 7 following days)

        dataframe_price_reports = pd.merge(left=dataframe_report_sorted_dates, right=dataframe_price, how='left',
                                           on='Released Date:')
        dataframe_price_reports = dataframe_price_reports.drop_duplicates()

        # Add new columns to the dataframe_price_reports table
        dataframe_price_reports = dataframe_price_reports.reindex(columns=dataframe_price_reports.columns.tolist() +
                                                                          ['Final open',
                                                                           'Final close',
                                                                           'Days after release until trading',
                                                                           '30 Day momentum',
                                                                           '180 Day momentum',
                                                                           '360 Day momentum',
                                                                           'Volatility'])

        dataframe_price_reports = get_final_close_prices(dataframe_price_reports)

        dataframe_price_reports = get_x_day_momentum(dataframe_price_reports)

        dataframe_price_reports = get_volatity(dataframe_price_reports)

        # drops all extraneous rows (retains only MD&A releases) and with valid auxiliary data
        dataframe_price_reports = dataframe_price_reports.dropna()

        dataframe_price_reports = get_returns(dataframe_price_reports)

        dataframe_price_reports = dataframe_price_reports[dataframe_price_reports['30 Day momentum'] != -90]
        dataframe_price_reports = dataframe_price_reports[dataframe_price_reports['180 Day momentum'] != -90]
        dataframe_price_reports = dataframe_price_reports[dataframe_price_reports['360 Day momentum'] != -90]
        # adds completed CSV to specified folder
        if not dataframe_price_reports.empty:
            dataframe_price_reports.to_csv(os.path.join(output_folder + price_filename), index=False)
    print("finished ",price_filename)

produce_csv_all_tickers(list_of_tickers)