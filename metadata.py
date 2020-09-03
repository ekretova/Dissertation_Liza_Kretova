import pandas as pd
import os
import pickle as pkl

project_directory = 'D:/dis/'
report_data_directory = 'D:/dis/report-pickle-data'
price_data_directory = 'D:/dis/price-csv-data'
price_output_file_name = 'company_price_metadata.csv'
mda_metadata_output_file_name = 'company_mda_metadata.csv'
ticker_complete_metadata_output_file_name = 'report_price_metadata.csv'

#Get the earliest and most recent dates of mda data
#Accomplished by taking first row and last row of an input ticker's mda pickle dataframe
#First row represents the latest (most recently recorded) date of mda data
#Last row represents the earliest (first available) date of mda data
def mda_dates_metadata(report_data_directory,mda_metadata_output_file_name):
    #Create placeholder dataframe to store columns required for metadata
    metadata_df = pd.DataFrame(columns = ['Ticker:', 'The earliest date of MD&A', 'The latest date of MD&A', 'Number of MD&A'])
    metadata_df.to_csv(mda_metadata_output_file_name, mode='w', header=True, index=False)

    #loop over all pickled report files extracting the first and last dates of MD&A creation
    for file in os.listdir(report_data_directory):
        #check file ends with .pickle (is correct type)
        if file.endswith('.pickle'):
            filename = os.fsdecode(file)
            print("currently processing mda data file of " + str(filename))
            #load file
            infile = open(os.path.join(report_data_directory, file),'rb')
            data = pkl.load(infile)
            #extract useful columns
            mda_dates = data[['Ticker:','Released Date:']]
            #check data is not empty
            if not mda_dates.empty:
                # Create a new temporary dataframe only containing the relevant dates
                mda_dates_temp = pd.concat([mda_dates.head(1), mda_dates.tail(1)])
                mda_dates_temp['The latest date of MD&A'] = mda_dates['Released Date:'].values[0]
                mda_dates_temp = mda_dates_temp.tail(1)
                # Rename columns to accurate reflect data and use for merging afterwards
                mda_dates_temp.rename({'Released Date:': 'The earliest date of MD&As'},axis=1, inplace=True)
                #New column for number of mdas taken from number of rows
                mda_dates_temp['Number of MD&A']= mda_dates.shape[0]
                mda_dates_temp.to_csv(mda_metadata_output_file_name, header=False, mode='a',index=False)
            infile.close()

#Get the earliest and most recent dates of price data
#Accomplished by taking first row and last row of an input ticker's price dataframe
#First row represents the earliest (first available) date of price data
#Last row represents the latest (most recently recorded) date of price data
#Returns a dataframe containing a single row with earliest and latest price data dates for the given input ticker dataframe
def price_dates_metadata(price_dataframe):
    #Create a new temporary dataframe only containing the relevant dates
    price_dates_dataframe = pd.concat([price_dataframe.head(1), price_dataframe.tail(1)])

    #Create a new column for earliest date, equal to the first value in the date column (earliest date)
    price_dates_dataframe['The earliest date of price'] = price_dates_dataframe['Date'].values[0]
    #Only keep the last row (which now contains the latest available date and the data mentioned above)
    price_dates_dataframe = price_dates_dataframe.tail(1)
    #Rename columns to accurate reflect data and use for merging afterwards
    price_dataframe=price_dates_dataframe.rename({'Date': 'The latest date of price'}, axis=1, inplace=True)
    price_dataframe=price_dates_dataframe.rename({'Symbol': 'Ticker:'}, axis=1, inplace=True)
    #Only keep relevant columns
    price_dataframe = price_dates_dataframe[['Ticker:', 'The earliest date of price', 'The latest date of price', 'Average daily trading volume in USD','Average daily trading volume in USD for 180 days starting 2 years ago']]
    #Format dates to be strings for further processing
    price_dates_dataframe['The earliest date of price'] = pd.to_datetime(price_dates_dataframe['The earliest date of price'].astype(str), format='%Y%m%d')
    price_dates_dataframe['The latest date of price'] = pd.to_datetime(price_dates_dataframe['The latest date of price'].astype(str), format='%Y%m%d')
    price_dataframe['The earliest date of price'] = price_dates_dataframe['The earliest date of price']
    price_dataframe['The latest date of price'] = price_dates_dataframe['The latest date of price']
    return price_dataframe

#Processes an input ticker's price dataframe to produce a new column with average daily trading volume
def average_trading_volume(price_dataframe):
    price_dataframe['Daily average trading volume'] = price_dataframe['Close'] * price_dataframe['Volume']
    total = price_dataframe['Daily average trading volume'].sum()
    total = round((total / price_dataframe.shape[0]), 3)
    price_dataframe['Average daily trading volume in USD'] = total
    return price_dataframe

#Produces average trading volume for 180 days, starting 2 years ago
def average_trading_volume_two_years_ago(price_dataframe):
    #Earliest date to be used is 180 days + 2 years ago
    #2 years ago = 2*262 (number of trading days in a year)
    reduced_price_dataframe = price_dataframe.tail(180 + 2 * 262)
    reduced_price_dataframe = price_dataframe.head(180)

    reduced_price_dataframe['Daily average trading volume'] = reduced_price_dataframe['Close'] * reduced_price_dataframe['Volume']
    total = reduced_price_dataframe['Daily average trading volume'].sum()
    total = round((total / reduced_price_dataframe.shape[0]), 3)
    price_dataframe['Average daily trading volume in USD for 180 days starting 2 years ago'] = total
    return price_dataframe

#When called with the input of destination directory, it will produce a finalised CSV output containing
#*The earliest date of price
#*The latest date of price
#*Average daily trading volume in USD
#*Average daily trading volume in USD for 180 days starting 2 years ago
#Must have all price data available in CSV format.
def price_data_metadata(price_data_directory):
    #create placeholder dataframe to store columsn required for the trading volume
    trading_volume_df = pd.DataFrame(columns = ['Ticker:', 'The earliest date of price','The latest date of price','Average daily trading volume in USD','Average daily trading volume in USD for 180 days starting 2 years ago'])
    trading_volume_df.to_csv(price_output_file_name, mode='w', header=True, index=False)
    #loop over files in given input directory
    for file in os.listdir(price_data_directory):
        #check if file is a csv
        if file.endswith('.csv'):
            #load file
            filename = os.fsdecode(file)
            print("currently processing price data of " + str(filename))
            company_price_dataframe=pd.read_csv(os.path.join(price_data_directory, file),usecols=['Symbol','Date','Close','Volume'])

            #Check file is not empty
            if not company_price_dataframe.empty:
                company_price_dataframe = average_trading_volume(company_price_dataframe)
                company_price_dataframe = average_trading_volume_two_years_ago(company_price_dataframe)

                company_price_dataframe = price_dates_metadata(company_price_dataframe)
                company_price_dataframe.to_csv(price_output_file_name, header=False, mode='a',index=False)

def ticker_complete_metadata(average_daily_volume_filter):
    #Merge price metadata and mda metadata for all tickers
    Table1 = pd.read_csv(os.path.join(project_directory, mda_metadata_output_file_name))
    Table2 = pd.read_csv(os.path.join(project_directory, price_output_file_name))
    f = pd.merge(left=Table1, right=Table2, how='inner', on='Ticker:')
    f = f.round({'Average daily trading volume in USD': 3})
    f = f.round({'Average daily trading volume in USD for 180 days starting 2 years ago': 3})
    df1 = f.dropna()

    #Remove all tickers that do not meet the required minimum average daily volume for trading
    df1 = df1[df1['Average daily trading volume in USD for 180 days starting 2 years ago'] > average_daily_volume_filter]

    #Remove any rows where the price data does not exist prior to the release of the earliest MD&A
    df1['The earliest date of MD&A'] = pd.to_datetime(df1['The earliest date of MD&A']).dt.strftime('%Y-%m-%d')
    df1 = df1[(df1['The earliest date of MD&A'] > df1['The earliest date of price'])]
    #Remove any rows wehre the price data does not exist after the latest MD&A
    df1['The latest date of MD&A'] = pd.to_datetime(df1['The latest date of MD&A']).dt.strftime('%Y-%m-%d')
    df1 = df1[(df1['The latest date of MD&A'] < df1['The latest date of price'])]
    #Sort metadata by average daily trading volumne
    df1 = df1.sort_values(by=['Average daily trading volume in USD'], ascending=False)

    # Set row for number of MD&A to equal number of rows-1 from price-report-output csv.
    # use column 'Number of MD&A'

    df1.to_csv('company_report_price_metadata.csv', index=False)

    print(df1.shape[0])


average_daily_volume_filter=200000

#Call defined functions
#mda_dates_metadata(report_data_directory,mda_metadata_output_file_name)
price_data_metadata(price_data_directory)
ticker_complete_metadata(average_daily_volume_filter)
