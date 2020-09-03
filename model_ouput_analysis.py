import pandas as pd

def model_accuracy(test_prediction,test_real,test_returns):
    test_set_df = pd.DataFrame()
    test_prediction_df = pd.DataFrame(test_prediction,columns = ['down','stay','up'])
    test_set_df['Real'] = test_real.idxmax(1)
    test_set_df['Preds'] =test_prediction_df.idxmax(1)

    print(test_real.idxmax(1).shape[0])
    print(test_returns.shape[0])
    print(test_returns.to_string())
    test_set_df['Return'] = test_returns
    percentage_real_positives = test_set_df[test_set_df['Real'] == 'up'].shape[0] / test_set_df.shape[0]

    labels_accs = {'true positive': 0, 'false positive': 0, 'errors': 0}
    for index, row in test_set_df.iterrows():
        if row['Real'] == 'up' and row['Preds'] == 'up':
            labels_accs['true positive'] += 1
        elif (row['Real'] == 'down' or row['Real'] == 'stay')and row['Preds'] == 'up':
            labels_accs['false positive'] += 1

    true_positive_percentage = round(((labels_accs['true positive'] / test_set_df.shape[0]) * 100), 2)
    false_positive_percentage = round(((labels_accs['false positive'] / test_set_df.shape[0]) * 100), 2)

    print('percentage of true positives' + str(true_positive_percentage) + '%')
    print('percentage of false positives' + str(false_positive_percentage) + '%')

    test_set_df.to_csv('embedding_aux_case_1_raw.csv',index=False)

    model_accuracy_metadata(true_positive_percentage,false_positive_percentage,percentage_real_positives)

    return ''

def model_accuracy_metadata(true_positive,false_positive,percentage_real_positives):

    model_accuracy_metadata = pd.DataFrame(columns = ['True Positives', 'False Positives', 'Total Positives', 'True Positives of All Positives','Ratio of real up movement'])
    total_positives = true_positive + false_positive
    model_accuracy_metadata = model_accuracy_metadata.append({'True Positives':true_positive,
                                                              'False Positives': false_positive,
                                                                'Total Positives': total_positives,
                                                                'True Positives of All Positives':(true_positive / total_positives),
                                                                'Ratio of real up movement':percentage_real_positives
                                                                }, ignore_index=True)

    print(model_accuracy_metadata.to_string())
    model_accuracy_metadata.to_csv('embedding_aux_model_cases.csv', mode='a',index=False)