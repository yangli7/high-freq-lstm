import pandas as pd
import numpy as np
import csv
import os


class DataSetGenerator:
    @staticmethod
    def generate_csv1(file_site, stock_id, date_index, step):
        """
        If the upper barrier is touched first, we label the observation as a 1.
        If the lower barrier is touched first, we label the observation as a -1.
        Otherwise, we label the observation as a 0.
        :param file_site: the exact directory of the csv file, eg.'H:\Wechat\WeChat Files\hhufjdd\Files\AI&FintechLab 2018 Material\SH600637'
        :param stock_id:
        :param date_index:
        :param step:
        :return:
        """
        # read file list
        file_list = os.listdir(file_site)

        f = open(file_site + '/' + file_list[date_index], 'r')
        raw_data = pd.read_csv(f)
        # delete the useless data
        raw_data = raw_data[raw_data.close > 0]

        price = raw_data.close.tolist()
        vol = raw_data.vol.tolist()
        # transform raw_data to the type list
        raw_data_list = (np.array(raw_data)).tolist()

        raw_data_length = len(raw_data)

        # create the csv file in which we are going to write the data
        csv_file = open('H:/sample_csv/' + stock_id + '/feature_and_label_of_' + file_list[date_index], 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(raw_data.columns.values.tolist() + ['5min', '10min', '20min'])

        sample_start = 0
        while sample_start + step + 400 + 5 < raw_data_length:
            y_start = sample_start + step
            total_vol_5min = raw_data.vol[y_start + 100 - 5:y_start + 100 + 6].sum()
            total_vol_10min = raw_data.vol[y_start + 200 - 5:y_start + 200 + 6].sum()
            total_vol_20min = raw_data.vol[y_start + 400 - 5:y_start + 400 + 6].sum()
            vmap_5min = 0
            vmap_10min = 0
            vmap_20min = 0
            if total_vol_5min == 0:
                vmap_5min = raw_data.price[y_start + 100 - 5:y_start + 100 + 6].mean()
            else:
                for ind in range(-5, 6):
                    vmap_5min += (vol[y_start + 100 + ind] / total_vol_5min) * price[y_start + 100 + ind]

            if total_vol_10min == 0:
                vmap_10min = raw_data.price[y_start + 200 - 5:y_start + 200 + 6].mean()
            else:
                for ind in range(-5, 6):
                    vmap_10min += (vol[y_start + 200 + ind] / total_vol_10min) * price[y_start + 200 + ind]

            if total_vol_20min == 0:
                vmap_20min = raw_data.price[y_start + 400 - 5:y_start + 400 + 6].mean()
            else:
                for ind in range(-5, 6):
                    vmap_20min += (vol[y_start + 400 + ind] / total_vol_20min) * price[y_start + 400 + ind]

            writer.writerow(raw_data_list[sample_start] + [vmap_5min, vmap_10min, vmap_20min])
            sample_start += step
        return







