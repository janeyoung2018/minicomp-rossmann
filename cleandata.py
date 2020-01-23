import holidays
import numpy as np
import pandas as pd

##i train set -> do whatever you want
##if test set -> use averages of the train set



def cleanData(df):
    # eliminate rows where store is empty
    df = df[~df.loc[:, 'Store'].isnull()]
    df.reset_index(inplace=True)

    print("Dropped rows without store-ids")

#    Join Function Needed here!

    # extract year, month and day from Date
    date = pd.DatetimeIndex(df.loc[:, 'Date'])
    df.loc[:, 'Year'] = date.year
    df.loc[:, 'Month'] = date.month
    df.loc[:, 'Day'] = date.day

    print("Extracted year, month and day from Date")

#   extract day of week
    df.loc[:, 'DayOfWeek'] = date.dayofweek + 1

    print("Extracted and reset day of week")

    for i in range(len(df)):
        if (np.isnan(df['Sales'][i])) & (df['Customers'][i] == 0):
            df['Sales'][i] = 0
        else:
            pass
    print('Set Sales to 0 if customers are 0')

#   deleting 0 sales rows
    df = df[df['Sales'] != 0]
    df.reset_index(inplace=True)
    print('Dropped 0-sales rows in df')

#   Sets Open to 1 if Sales happened while Open is 0

    for i in range(len(df)):
        if (df['Sales'][i] > 0) & (np.isnan(df['Open'][i])):
            df['Open'][i] = 1
        else:
            pass

    print('Set Open = 1 if Sales > 0')

#   function to fill school holiday based on state holiday

    def helper_schoolholiday(row):
        for i in range(len(df)):
            if pd.isnull(df['SchoolHoliday'][i]):
                return 0.0
            else:
                return df['SchoolHoliday'][i]

    df['SchoolHoliday'] = df.apply(helper_schoolholiday, axis=1)
    print("Filled school holidays based on state holidays")

# #   Taking care of shops in train stations
#     def train_station_stores_nan_open(df):
#         mask = df.loc[:, 'DayOfWeek'] == 7.0
#         dftrain2 = df[mask]
#         dftrain3 = dftrain2.groupby('Store')['Open'].sum().to_frame().rename(columns={'Open': 'newopen'})
#         train_station_stores = [i for i in dftrain3[(dftrain3.newopen > 3)].index]
#
#         if (pd.isnull(row['Open'])) & (row['Store'] in train_station_stores):
#             return 1.0
#         else:
#             return row['Open']
#
#     df['Open'] = df.apply(train_station_stores_nan_open, axis=1)
#
#     print("Train station stores now open! Enjoy your sunday shopping!")

    # """Sets all Shops with isna('Open') to 0 on a German public holiday"""
    #de_holidays = holidays.DE()

    for i in range(len(df)):
        if (np.isnan(df['Open'][i])) & (df['Date'][i] in holidays.DE()):
            df['Open'][i] = 0
        else:
            pass
    print('Public Holidays updated')

    # take care of regional stateholiday
    for i in range(len(df)):
        if (pd.isnull(df['StateHoliday'][i])) & (df['Month'][i] == 1) & (df['Day'][i] == 6):
            if df['Year'][i] == 2013:
                storename = df['Store'][i]
                row1 = df[df.Store == storename]
                row2 = row1[row1.Date == '2014-01-06']
                try:
                    df['StateHoliday'][i] = row2['StateHoliday'].values[0]
                except:
                    pass
            else:
                storename = df['Store'][i]
                row1 = df[df.Store == storename]
                row2 = row1[row1.Date == '2013-01-06']
                try:
                    train['StateHoliday'][i] = row2['StateHoliday'].values[0]
                except:
                    pass
        elif (pd.isnull(df['StateHoliday'][i])) & (df['Month'][i] == 6) & (df['Day'][i] == 1):
            if df['Year'][i] == 2013:
                storename = df['Store'][i]
                row1 = df[df.Store == storename]
                row2 = row1[row1.Date == '2014-06-01']
                try:
                    df['StateHoliday'][i] = row2['StateHoliday'].values[0]
                except:
                    pass
            else:
                storename = df['Store'][i]
                row1 = df[df.Store == storename]
                row2 = row1[row1.Date == '2013-06-01']
                try:
                    df['StateHoliday'][i] = row2['StateHoliday'].values[0]
                except:
                    pass
        else:
            pass

    print('Finished regional stateholidays')

    # take care of remained stateholiday
    def remained_stateholiday(row):
        if (pd.isnull(row['StateHoliday'])):
            if pd.isnull(row['Open']):
                if row['Sales'] > 0:
                    return '0'
                else:
                    return 'a'
            else:
                if row['Open'] == 0.0:
                    return 'a'
                else:
                    return '0'
        else:
            return row['StateHoliday']

    df['StateHoliday'] = df.apply(remained_stateholiday, axis=1)
    print('finished remaining stateholidays')

    # Sets all Shops with isna('Open') to 0 based on stateholiday state
    def open_stateholiday(row):
        if pd.isnull(row['Open']) & (row['StateHoliday'] == 'a') & (row['StateHoliday'] == 'b') & (
                row['StateHoliday'] == 'c'):
            return 0.0
        elif pd.isnull(row['Open']) & (row['StateHoliday'] == '0'):
            return 1.0
        else:
            return row['Open']

    df['Open'] = df.apply(open_stateholiday, axis=1)

    print('Adjusted open status of shops according to state holidays')

# fill empty 'Customers' with average customer number when open=1.0, when open=0.0 customer=0.0
    df_mean_customers = df.loc[:, 'Customers'].mean()

    def helper_customers(row):
        if (pd.isnull(row['Customers'])) & (row['Open'] == 1.0):
            return df_mean_customers
        elif (pd.isnull(row['Customers'])) & (row['Open'] == 0.0):
            return 0.0
        else:
            return row['Customers']

    df.loc[:, 'Customers'] = df.apply(helper_customers, axis=1)

    print('Finished filling in empty customers cells')



    # Fills empty 'Sales'-Cells in train with average if there have been non 0 customers in the shop
    mean_sales = df['Sales'].mean()

    def helper_sales(row):
        if pd.isnull(row['Sales']) & (row['Customers'] > 0):
            return mean_sales
        else:
            return row['Sales']

    df['Sales'] = df.apply(helper_sales, axis=1)
    print("Finish sales")

    #Gets dummies for 'StateHolidays' into three columns and concat them to the table
    NewStateHoliday = pd.get_dummies(df['StateHoliday'])
    NewStateHoliday.rename(columns={'0': 'NoStateHoliday', 'a': 'PublicHoliday', 'b': 'EasterHoliday', 'c': 'Christmas Holiday'}, inplace=True)
    df = pd.concat([df, NewStateHoliday], axis=1)
    print('State Holidays Encoded')

    #Gets dummies for 'PromoInterval' into three columns and concat them to the table
    PromoInterval = pd.get_dummies(df['PromoInterval'])
    df = pd.concat([df, PromoInterval], axis=1)
    print('PromoIntervals encoded')


    def fillEmptyDistances(row):
        """Filling empty distances with mean"""
        if pd.isnull(row['CompetitionDistance']):
            return df['CompetitionDistance'].mean()
        else:
            return row['CompetitionDistance']

    df['CompetitionDistance'] = df.apply(fillEmptyDistances, axis=1)

    #Encoding store types
    NewStoreType = pd.get_dummies(df['StoreType'])
    NewStoreType.rename(columns={'a': 'StoreType a', 'b': 'StoreType b', 'c': 'StoreType c', 'd': 'StoreType d'},
                        inplace=True)
    store = pd.concat([df, NewStoreType], axis=1)
    print('Store Type Encoded')


    #Gets dummies for 'Assortment' into three columns and concat them to the table
    NewAssortment = pd.get_dummies(store['Assortment'])
    NewAssortment.rename(columns={'a': 'Basic Assort', 'b': 'Extra Assort', 'c': 'Extended Assort'}, inplace=True)
    store = pd.concat([df, NewAssortment], axis=1)
    print('Assortment Type Encoded')
    print('---Cleaning completed---')