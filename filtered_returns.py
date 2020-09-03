import pickle

import pandas as pd
import os
import pickle as pkl



report_data_directory = 'D:/dis/report-pickle-data'
price_report_data_directory = 'D:/dis/price-report-data-200000/'
output_folder = 'D:/dis/final-preprocessing-pickle-data/'
returns_folder = 'D:/dis/return-filtered-table/'

#loop over all processed price CSVs for the MD&&A dates and combine into one file
def get_csv_all_price_aux_data():
    return_value_filter_list=pd.DataFrame()
    for file in os.listdir(price_report_data_directory):
        return_value_filter_list = return_value_filter_list.append(pd.read_csv(price_report_data_directory + file))
    return_value_filter_list.to_csv(os.path.join(returns_folder + 'all_returns-200000.csv'), index=False)


#Create a dataframe to store price values for price & auxiliary data on the release dates of MD&As for all companies
def processed_mda_price_merge():

    #create dataframe for final values (outside any loop)
    mda_auxiliary_full = pd.DataFrame(columns = ['Ticker:','Released Date:','Released Time','Open','Close','Final open',
                                                                           'Final close',
                                                                           'Days after release until trading',
                                                                           '30 Day momentum',
                                                                           '180 Day momentum',
                                                                           '360 Day momentum',
                                                                           'Volatility','Return','Processed Text:'])

    dataframe_filtered = pd.read_csv(os.path.join(returns_folder, 'all_returns.csv'))

    #Loops over all company MD&A pickle files
    for file in os.listdir(report_data_directory):
        if file.endswith('.pickle'):
            filename = os.fsdecode(file)
            print(filename)
            #load the dataframe with MD&A info
            infile = open(os.path.join(report_data_directory, file),'rb')
            data = pkl.load(infile)
            #only need relevant columns
            df = data[['Ticker:','Released Date:','Processed Text:']]
            #only attempt to process the datetime value into separate date and time columns if dataframe is not empty
            if not df.empty:
                df['Released Time'] = pd.to_datetime(df['Released Date:']).dt.strftime(
                    '%H:%M:%S')
                df['Released Date:'] = pd.to_datetime(df['Released Date:']).dt.strftime(
                    '%Y-%m-%d')
                #df['Released Date:'] = pd.to_datetime(df['Released Date:'].astype(str),
                                                                   # format='%Y-%m-%d')

                #merge the MD&A dataframe with the true positive returns dataframe
                return_filtered_mda = pd.merge(left=dataframe_filtered, right=df, how='inner',
                                               on=['Released Time','Ticker:','Released Date:'])
            # print(df.to_string())
                print(return_filtered_mda.to_string())
            #dataframe_filtered_mda.to_csv(os.path.join(output_folder + 'test.csv'), index=False)
                mda_auxiliary_full=mda_auxiliary_full.append(return_filtered_mda)
                print(mda_auxiliary_full.to_string())

    mda_auxiliary_full = mda_auxiliary_full.reset_index(drop=True)

    return mda_auxiliary_full

#
def stock_movements(preprocessed_mdas, return_percentage):
    # get distribution of returns that are up / down / remain
    # previous value 0.74
    stock_movements = []
    for index, row in preprocessed_mdas.iterrows():
        if row['Return'] >= return_percentage:
            stock_movements.append('up')
        elif row['Return'] <= -(return_percentage):
            stock_movements.append('down')
        else:
            stock_movements.append('stay')

    preprocessed_mdas['Stock Price Movements:'] = stock_movements

    seriesObj1 = preprocessed_mdas.apply(lambda x: "up" if x['Stock Price Movements:'] == 'up' else False, axis=1)
    seriesObj2 = preprocessed_mdas.apply(lambda x: "down" if x['Stock Price Movements:'] == 'down' else False, axis=1)
    seriesObj3 = preprocessed_mdas.apply(lambda x: "stay" if x['Stock Price Movements:'] == 'stay' else False, axis=1)

    # Count number of True in series
    numOfRowsUp = len(seriesObj1[seriesObj1 == "up"].index)
    numOfRowsDown = len(seriesObj2[seriesObj2 == "down"].index)
    numOfRowsStay = len(seriesObj3[seriesObj3 == "stay"].index)
    print('up : ', numOfRowsUp)
    print('down : ', numOfRowsDown)
    print('stay : ', numOfRowsStay)

    return preprocessed_mdas

#get_csv_all_price_aux_data()
#processed_mdas_auxiliary=processed_mda_price_merge()

prepocessed_data_directory = 'D:/dis/final-preprocessing-pickle-data'
preprocessed_file_name = 'mda_aux_df5.pickle'
output_folder = 'D:/dis/finbert-modelling/'
infile = open(os.path.join(prepocessed_data_directory, preprocessed_file_name), 'rb')
preprocessed_mdas = pickle.load(infile)

processed_mdas_auxiliary=stock_movements(preprocessed_mdas, 0.74)
print(processed_mdas_auxiliary.to_string())

with open(prepocessed_data_directory+'mda_aux_df_movement.pickle', 'wb') as f:
    pkl.dump(processed_mdas_auxiliary, f)
