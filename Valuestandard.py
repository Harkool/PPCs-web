from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class Valuestandard:

    def __init__(self, skip_list=None, onehot_list=None, categories_list=None, Skip=False, OneHot=False):

        self.__Skip = Skip
        self.__OneHot = OneHot

        if self.__Skip:
            if type(skip_list) == type(None):
                raise ValueError("Parameter 'skip_list' is required.")
            elif type(skip_list) != type([]):
                raise ValueError("Parameter 'skip_list' must be of type list.")
            else:
                self.__list1 = skip_list

        if self.__OneHot:
            if type(onehot_list) == type(None):
                raise ValueError("Parameter 'onehot_list' is required.")
            elif type(onehot_list) != type([]):
                raise ValueError("Parameter 'onehot_list' must be of type list.")
            else:
                self.__list2 = onehot_list

        if self.__OneHot:
            if type(categories_list) == type(None):
                raise ValueError("Parameter 'categories_list' is required.")
            elif type(categories_list) != type([]):
                raise ValueError("Parameter 'categories_list' must be of type list.")
            else:
                for self.__k in range(len(categories_list)):
                    if self.__k == 0:
                        self.__list3 = [list(range(categories_list[self.__k]))]
                    else:
                        self.__list3.append(list(range(categories_list[self.__k])))

        self.__fitstate = False
        self.__fitstate_2 = False

        return None

    def fit_transform(self, data_1=None):
        if self.__fitstate:
            raise RuntimeError("Data already standardized. Reinitialize to standardize again.")

        if type(data_1) == None:
            raise ValueError("Parameter 'data_1' is required.")
        elif type(data_1) != type(pd.DataFrame()):
            raise ValueError("Input must be a DataFrame, got %s" % type(data_1))
        else:
            self.__Data_1 = data_1.copy()
            self.__data = self.__Data_1.copy()

        if self.__OneHot:
            self.__Oname_list = self.__data.columns[self.__list2]
            self.__data_1 = self.__data[self.__Oname_list].copy()
            self.__data = self.__data.drop(self.__Oname_list, axis=1)

            for self.__i in range(len(self.__list2)):
                self.__transdata = self.__data_1[self.__Oname_list[self.__i]].copy()
                self.__quantity = self.__list3[self.__i]
                self.__LE = LabelEncoder()
                self.__transdata = self.__LE.fit_transform(self.__transdata)
                self.__OH = OneHotEncoder(categories=[self.__quantity], sparse=False)
                self.__transdata = self.__OH.fit_transform(self.__transdata.reshape(-1, 1))
                self.__transdata = pd.DataFrame(self.__transdata)
                self.__transdata.rename(columns=lambda s: self.__Oname_list[self.__i] + '_' + str(s), inplace=True)
                self.__transdata = self.__transdata.astype(np.int64)

                if self.__i == 0:
                    self.__data_onehot = self.__transdata.copy()
                else:
                    self.__data_onehot = pd.concat([self.__data_onehot, self.__transdata], axis=1)

        if self.__Skip:
            self.__Sname_list = self.__Data_1.columns[self.__list1]
            self.__data_skip = self.__data[self.__Sname_list].copy()
            self.__data_o2 = self.__data[self.__Sname_list].copy()
            self.__data = self.__data.drop(self.__Sname_list, axis=1)
            self.__s = self.__data.shape[1] != 0
        else:
            self.__s = True

        if self.__s:
            self.__Dname_list = self.__data.columns
            self.__data_original = self.__data.copy()
            self.__Key = StandardScaler()
            self.__data_standard = self.__Key.fit_transform(self.__data)
            self.__data_standard = pd.DataFrame(self.__data_standard)
            self.__data_standard.columns = self.__Dname_list
        else:
            self.__data_standard = self.__data.copy()
            self.__data_original = self.__data.copy()

        if self.__Skip:
            self.__data_standard = pd.concat([self.__data_standard, self.__data_skip], axis=1)
            self.__data_original = pd.concat([self.__data_original, self.__data_o2], axis=1)

        if self.__OneHot:
            self.__data_standard = pd.concat([self.__data_standard, self.__data_onehot], axis=1)
            self.__data_original = pd.concat([self.__data_original, self.__data_onehot], axis=1)

        self.__fitstate = True
        return None

    def transform(self, data_2=None):
        if not self.__fitstate:
            raise RuntimeError("Call fit_transform() before transform().")

        if type(data_2) == None:
            raise ValueError("Parameter 'data_2' is required.")
        elif type(data_2) != type(pd.DataFrame()):
            raise ValueError("Input must be a DataFrame, got %s" % type(data_2))
        else:
            self.__Data_2 = data_2.copy()
            self.__data = self.__Data_2.copy()

        if self.__OneHot:
            self.__Oname_list = self.__data.columns[self.__list2]
            self.__data_1 = self.__data[self.__Oname_list].copy()
            self.__data = self.__data.drop(self.__Oname_list, axis=1)

            for self.__i in range(len(self.__list2)):
                self.__transdata = self.__data_1[self.__Oname_list[self.__i]].copy()
                self.__quantity = self.__list3[self.__i]
                self.__LE = LabelEncoder()
                self.__transdata = self.__LE.fit_transform(self.__transdata)
                self.__OH = OneHotEncoder(categories=[self.__quantity], sparse=False)
                self.__transdata = self.__OH.fit_transform(self.__transdata.reshape(-1, 1))
                self.__transdata = pd.DataFrame(self.__transdata)
                self.__transdata.rename(columns=lambda s: self.__Oname_list[self.__i] + '_' + str(s), inplace=True)
                self.__transdata = self.__transdata.astype(np.int64)

                if self.__i == 0:
                    self.__data_onehot = self.__transdata.copy()
                else:
                    self.__data_onehot = pd.concat([self.__data_onehot, self.__transdata], axis=1)

        if self.__Skip:
            self.__Sname_list = self.__Data_2.columns[self.__list1]
            self.__data_skip = self.__data[self.__Sname_list].copy()
            self.__data_o2 = self.__data[self.__Sname_list].copy()
            self.__data = self.__data.drop(self.__Sname_list, axis=1)
            self.__st = self.__data.shape[1] != 0
        else:
            self.__st = True

        if self.__st:
            self.__Dname_list = self.__data.columns
            self.__data_original_2 = self.__data.copy()
            self.__data_standard_2 = self.__Key.transform(self.__data)
            self.__data_standard_2 = pd.DataFrame(self.__data_standard_2)
            self.__data_standard_2.columns = self.__Dname_list
        else:
            self.__data_standard_2 = self.__data.copy()
            self.__data_original_2 = self.__data.copy()

        if self.__Skip:
            self.__data_standard_2 = pd.concat([self.__data_standard_2, self.__data_skip], axis=1)
            self.__data_original_2 = pd.concat([self.__data_original_2, self.__data_o2], axis=1)

        if self.__OneHot:
            self.__data_standard_2 = pd.concat([self.__data_standard_2, self.__data_onehot], axis=1)
            self.__data_original_2 = pd.concat([self.__data_original_2, self.__data_onehot], axis=1)

        self.__fitstate_2 = True
        return None

    def standard_data_1(self):
        if not self.__fitstate:
            raise RuntimeError("Call fit_transform() first.")
        return self.__data_standard

    def standard_data_2(self):
        if not self.__fitstate_2:
            raise RuntimeError("Call transform() first.")
        return self.__data_standard_2

    def original_data_1(self):
        if not self.__fitstate:
            raise RuntimeError("Call fit_transform() first.")
        return self.__data_original

    def original_data_2(self):
        if not self.__fitstate_2:
            raise RuntimeError("Call transform() first.")
        return self.__data_original_2
