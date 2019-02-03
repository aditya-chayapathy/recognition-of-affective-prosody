import pandas as pd
import scipy
import scipy.io
import scipy.signal
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class PeakDetection(object):

    def __init__(self):
        self.input_path = "../resources/input"
        self.test_data_location = '../resources/input/test'
        self.train_expected_location = "../resources/input/train/expected"
        self.train_unexpected_location = "../resources/input/train/unexpected"
        self.test_expected_location = "../resources/input/test/expected"
        self.test_unexpected_location = "../resources/input/test/unexpected"
        self.train_expected_file_list = [os.path.join(self.train_expected_location, f) for f in os.listdir(self.train_expected_location) if f.endswith('.mat')]
        self.train_unexpected_file_list = [os.path.join(self.train_unexpected_location, f) for f in os.listdir(self.train_unexpected_location) if f.endswith('.mat')]
        self.test_data_file_list =[os.path.join(self.test_data_location, f) for f in os.listdir(self.test_data_location) if f.endswith('.mat')]
        self.train_expected_data = self.setup_data(self.train_expected_file_list, "expected", True)
        self.train_unexpected_data = self.setup_data(self.train_unexpected_file_list, "unexpected", True)
        self.test_expected_file_list = [os.path.join(self.test_expected_location, f) for f in
                                         os.listdir(self.test_expected_location) if f.endswith('.mat')]
        self.test_unexpected_file_list = [os.path.join(self.test_unexpected_location, f) for f in
                                           os.listdir(self.test_unexpected_location) if f.endswith('.mat')]
        self.test_expected_data = self.setup_data(self.test_expected_file_list, "expected", True)
        self.test_unexpected_data = self.setup_data(self.test_unexpected_file_list, "unexpected", True)
        self.train_data = self.train_expected_data.append(self.train_unexpected_data)
        self.test_data = self.test_expected_data.append(self.test_unexpected_data)

    def setup_data(self, file_list, label, filter = True):
        final_df = pd.DataFrame()
        for each in file_list:
            mat = scipy.io.loadmat(each)
            df = pd.DataFrame(mat['data'])
            df = df.drop(df.columns[1], axis = 1)
            if filter:
                df = df[350:2200]
            df_ = df.T
            final_df = final_df.append(df_)
        final_df['label'] = label
        return final_df

    def svm_on_raw_data(self):
        expected_train_df = self.setup_data(self.train_expected_file_list, "expected", True)
        unexpected_train_df = self.setup_data(self.train_expected_file_list, "unexpected", True)
        train_df = expected_train_df.append(unexpected_train_df)
        train_list = train_df.loc[:, train_df.columns != 'label'].values
        label_list = train_df['label'].tolist()
        clf = SVC(kernel='rbf')
        clf.fit(train_list, label_list)
        test_expected_df = self.setup_data(self.test_expected_file_list, "expected", True)
        test_unexpected_df = self.setup_data(self.test_unexpected_file_list, "unexpected", True)
        test_df = test_expected_df.append(test_unexpected_df)
        test_list = test_df.loc[:, test_df.columns != 'label'].values
        test_label_list = test_df['label'].tolist()
        right_count = 0
        for i in range(len(test_list)):
            pred = clf.predict([test_list[i]])
            if pred[0] == test_label_list[i]:
                right_count += 1
        print("SVM on raw data: " + str(right_count / len(test_list) * 100))

    
    def knn_on_raw_data(self):
        expected_train_df = self.setup_data(self.train_expected_file_list, "expected", True)
        unexpected_train_df = self.setup_data(self.train_expected_file_list, "unexpected", True)
        train_df = expected_train_df.append(unexpected_train_df)
        train_list = train_df.loc[:, train_df.columns != 'label'].values
        label_list = train_df['label'].tolist()
        clf = KNeighborsClassifier(n_neighbors=4)
        clf.fit(train_list, label_list)
        test_expected_df = self.setup_data(self.test_expected_file_list, "expected", True)
        test_unexpected_df = self.setup_data(self.test_unexpected_file_list, "unexpected", True)
        test_df = test_expected_df.append(test_unexpected_df)
        test_list = test_df.loc[:, test_df.columns != 'label'].values
        test_label_list = test_df['label'].tolist()
        right_count = 0
        for i in range(len(test_list)):
            pred = clf.predict([test_list[i]])
            if pred[0] == test_label_list[i]:
                right_count += 1
        print("KNN on raw data: " + str(right_count / len(test_list) * 100))

    def peak_detection_setup(self, type ='Postive'):
        peak_list = []
        train_list = self.train_data.loc[:, self.train_data.columns != 'label'].values
        label_list = self.train_data['label'].tolist()
        for i in range(len(train_list)):
            peak_input = train_list[i]
            if type == 'Negative':
                peak_input = [-x for x in peak_input]
            peaks = scipy.signal.find_peaks(peak_input)
            peak_list.append(list(peaks[0]))
        min_peak_len = 222
        for each in peak_list:
            min_peak_len = len(each) if len(each) < min_peak_len else min_peak_len
        for i in range(len(peak_list)):
            peak_list[i] = peak_list[i][:min_peak_len]
        test_list = self.test_data.loc[:, self.test_data.columns != 'label'].values
        test_label_list = self.test_data['label'].tolist()
        return peak_list, label_list, test_list, test_label_list

    def positive_peak_detection_knn(self):
        train_peak_list, train_label_list, test_list, test_label_list = self.peak_detection_setup("Positive")
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_peak_list, train_label_list)
        right_count = 0
        min_peak_len = len(train_peak_list[0])
        for i in range(len(test_list)):
            peak = scipy.signal.find_peaks(test_list[i])[0]
            pred = clf.predict([peak[:min_peak_len].tolist()])
            if pred[0] == test_label_list[i]:
                right_count += 1
        print("Positive Peak Detection (P300) KNN: " + str(right_count / len(test_list) * 100))

    def positive_peak_detection_SVM(self):
        train_peak_list, train_label_list, test_list, test_label_list = self.peak_detection_setup("Positive")
        clf = SVC(kernel='rbf')
        clf.fit(train_peak_list, train_label_list)
        right_count = 0
        min_peak_len = len(train_peak_list[0])
        for i in range(len(test_list)):
            peak = scipy.signal.find_peaks(test_list[i])[0]
            pred = clf.predict([peak[:min_peak_len].tolist()])
            if pred[0] == test_label_list[i]:
                right_count+=1
        print("Positive Peak Detection (P300) SVM: " + str(right_count/len(test_list)*100))


    def negative_peak_detection_KNN(self):
        train_peak_list, train_label_list, test_list, test_label_list = self.peak_detection_setup("Negative")
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_peak_list, train_label_list)
        right_count = 0
        min_peak_len = len(train_peak_list[0])
        for i in range(len(test_list)):
            neg_arr = [-x for x in test_list[i]]
            peak = scipy.signal.find_peaks(neg_arr)[0]
            pred = clf.predict([peak[:min_peak_len].tolist()])
            if pred[0] == test_label_list[i]:
                right_count += 1
        print("Negative Peak Detection (N300) KNN: " + str(right_count/len(test_list)*100))

    def negative_peak_detection_SVM(self):
        train_peak_list, train_label_list, test_list, test_label_list = self.peak_detection_setup("Negative")
        clf = SVC(kernel='rbf')
        clf.fit(train_peak_list, train_label_list)
        right_count = 0
        min_peak_len = len(train_peak_list[0])
        for i in range(len(test_list)):
            neg_arr = [-x for x in test_list[i]]
            peak = scipy.signal.find_peaks(neg_arr)[0]
            pred = clf.predict([peak[:min_peak_len].tolist()])
            if pred[0] == test_label_list[i]:
                right_count += 1
        print("Negative Peak Detection (N300) SVM: " + str(right_count / len(test_list) * 100))

    def negative_peak_detection_avg_setup(self):
        train_list = self.train_data.loc[:, self.train_data.columns!='label'].values
        label_list = self.train_data['label'].tolist()
        expected_peak_index_list = []
        unexpected_peak_index_list = []
        for i in range(len(train_list)):
            neg_arr = [-x for x in train_list[i]]
            peak = neg_arr.index(max(neg_arr))
            if label_list[i] == 'expected':
                expected_peak_index_list.append(peak+350)
            else:
                unexpected_peak_index_list.append(peak+350)
        return sum(expected_peak_index_list)/len(expected_peak_index_list), sum(unexpected_peak_index_list)/len(unexpected_peak_index_list)

    def negative_peak_using_time_avg(self):
        correct_count = 0
        expected_peak_avg_time, unexpected_peak_avg_time = self.negative_peak_detection_avg_setup()
        train_list = self.test_data.loc[:, self.test_data.columns != 'label'].values
        label_list = self.test_data['label'].tolist()
        predicted_label_list =['']*len(label_list)
        for i in range(len(train_list)):
            neg_arr = [-x for x in train_list[i]]
            peak = neg_arr.index(max(neg_arr))
            if abs(peak - expected_peak_avg_time) >= abs(peak - unexpected_peak_avg_time):
                predicted_label_list[i] = 'expected'
            else:
                predicted_label_list[i] = 'unexpected'
        for i in range(len(label_list)):
            if predicted_label_list[i] == label_list[i]:
                correct_count += 1
        pd.DataFrame(list(zip(label_list, predicted_label_list)), columns=['Actual', 'Predicted']).to_csv('N300_output.csv',index=False)
        print("Negative Peak detection (N300) using peak avg time: " + str(correct_count/len(label_list)*100))


if __name__ == "__main__":
    PeakDetector = PeakDetection()
    PeakDetector.negative_peak_using_time_avg()
    PeakDetector.positive_peak_detection_knn()
    PeakDetector.negative_peak_detection_KNN()
    PeakDetector.positive_peak_detection_SVM()
    PeakDetector.negative_peak_detection_SVM()
    PeakDetector.knn_on_raw_data()
    PeakDetector.svm_on_raw_data()