Matlab code:
1) Open this folder in MATLAB
2) ExpectedClassification_RawData: Uses raw training data to build the model
3)ExpectedClassification_MovMedian: Uses Moving Median Low pass filter on the raw data to build the model .
3) ExpectedClassification_DWT: Uses DWT on the raw data to build the model.

Python code:
1) Written in Python3
2) Run the peak_detection.py to run all the models that we have worked on:
	a) KNN and SVM on Raw Data
	b) Classification using P300 peak detection (KNN and SVM)
	c) Classification using N300 peak detection (Avg Negative Peak Time Window Analysis, KNN and SVM)
