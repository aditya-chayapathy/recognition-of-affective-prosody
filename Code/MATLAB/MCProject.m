%MC Project
training_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Classification\Training\";
testing_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Classification\Testing\";
output_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Output\Class\Subject 1\";

%Creating output directory if it doesn't exist
if ~exist(output_folder_path, 'dir')
    mkdir(char(output_folder_path));
end

modified_file_path = strcat(training_folder_path, "\**\*.mat");
dir_info = dir(char(modified_file_path));
noisy_training_data_table = cell2table({});
noiseless_training_data_table = cell2table({});
dwt_training_data_table = cell2table({});
training_class_labels_table = cell2table({});
for K = 1:length(dir_info)
    sub_dir_file_name = dir_info(K).name;
    sub_dir_folder = dir_info(K).folder;
    fileName = strcat(sub_dir_folder, "\", sub_dir_file_name);
    file_name = erase(sub_dir_file_name, ".mat");
    cells = strsplit(sub_dir_folder, "\");
    class_name = cells{length(cells)};
    data_structure = load(fileName);
    data = data_structure.data;
    data_table = array2table(data);
    
    noisy_data_table = data_table(1:end, 1:1);
    noiseless_data_array = movmedian(table2array(noisy_data_table), 200);
    dwt_features_data_array = dwt(noiseless_data_array, 'sym4');

    
    noisy_data_table_transpose = array2table(table2array(noisy_data_table).');
    noisy_training_data_table = [noisy_training_data_table; noisy_data_table_transpose];
    
    noiseless_data_table_transpose = array2table(noiseless_data_array.');
    noiseless_training_data_table = [noiseless_training_data_table; noiseless_data_table_transpose];
    
    dwt_data_table_transpose = array2table(dwt_features_data_array.');
    dwt_training_data_table = [dwt_training_data_table; dwt_data_table_transpose];
    
    training_class_labels_table = [training_class_labels_table; cell2table({class_name})];
end

%Training classification models
noisy_data_SVM_model = fitcsvm(noisy_training_data_table, training_class_labels_table);
noisy_data_DT_model = fitctree(noisy_training_data_table, training_class_labels_table);
noisy_data_KNN_model = fitcknn(noisy_training_data_table, training_class_labels_table);

noiseless_data_SVM_model = fitcsvm(noiseless_training_data_table, training_class_labels_table);
noiseless_data_DT_model = fitctree(noiseless_training_data_table, training_class_labels_table);
noiseless_data_KNN_model = fitcknn(noiseless_training_data_table, training_class_labels_table);

dwt_data_SVM_model = fitcsvm(dwt_training_data_table, training_class_labels_table);
dwt_data_DT_model = fitctree(dwt_training_data_table, training_class_labels_table);
dwt_data_KNN_model = fitcknn(dwt_training_data_table, training_class_labels_table);


modified_file_path = strcat(testing_folder_path, "\**\*.mat");
dir_info = dir(char(modified_file_path));

noisy_svm_testing_results = cell2table({});
noisy_dt_testing_results = cell2table({});
noisy_knn_testing_results = cell2table({});
noiseless_svm_testing_results = cell2table({});
noiseless_dt_testing_results = cell2table({});
noiseless_knn_testing_results = cell2table({});
dwt_svm_testing_results = cell2table({});
dwt_dt_testing_results = cell2table({});
dwt_knn_testing_results = cell2table({});

for K = 1:length(dir_info)
    sub_dir_file_name = dir_info(K).name;
    sub_dir_folder = dir_info(K).folder;
    fileName = strcat(sub_dir_folder, "\", sub_dir_file_name);
    file_name = erase(sub_dir_file_name, ".mat");
    cells = strsplit(sub_dir_folder, "\");
    class_name = cells{length(cells)};
    data_structure = load(fileName);
    data = data_structure.data;
    data_table = array2table(data);
    noisy_data_table = data_table(1:end, 1:1);
    table_array = table2array(noisy_data_table);
    noisy_data_table_transpose = array2table(table_array.');
    filename_table = cell2table({file_name});
    filename_table.Properties.VariableNames = {'FileName'};
    
    %Noisy Data SVM
    noisy_svm_class_label = predict(noisy_data_SVM_model, noisy_data_table_transpose);
    noisy_svm_class_label_table = cell2table({noisy_svm_class_label});
    noisy_svm_class_label_table.Properties.VariableNames = {'ClassLabel'};
    noisy_svm_temp_table = [filename_table noisy_svm_class_label_table];
    noisy_svm_testing_results = [noisy_svm_testing_results; noisy_svm_temp_table];
    
    %Noisy Data DT
    noisy_dt_class_label = predict(noisy_data_DT_model, noisy_data_table_transpose);
    noisy_dt_class_label_table = cell2table({noisy_dt_class_label});
    noisy_dt_class_label_table.Properties.VariableNames = {'ClassLabel'};
    noisy_dt_temp_table = [filename_table noisy_dt_class_label_table];
    noisy_dt_testing_results = [noisy_dt_testing_results; noisy_dt_temp_table];
    
    %Noisy Data KNN
    noisy_knn_class_label = predict(noisy_data_KNN_model, noisy_data_table_transpose);
    noisy_knn_class_label_table = cell2table({noisy_knn_class_label});
    noisy_knn_class_label_table.Properties.VariableNames = {'ClassLabel'};
    noisy_knn_temp_table = [filename_table noisy_knn_class_label_table];
    noisy_knn_testing_results = [noisy_knn_testing_results; noisy_knn_temp_table];
    
    
    
    %Noiseless Data SVM
    noiseless_svm_class_label = predict(noiseless_data_SVM_model, noiseless_data_table_transpose);
    noiseless_svm_class_label_table = cell2table({noiseless_svm_class_label});
    noiseless_svm_class_label_table.Properties.VariableNames = {'ClassLabel'};
    noiseless_svm_temp_table = [filename_table noiseless_svm_class_label_table];
    noiseless_svm_testing_results = [noiseless_svm_testing_results; noiseless_svm_temp_table];
    
    %Noiseless Data DT
    noiseless_dt_class_label = predict(noiseless_data_DT_model, noiseless_data_table_transpose);
    noiseless_dt_class_label_table = cell2table({noiseless_dt_class_label});
    noiseless_dt_class_label_table.Properties.VariableNames = {'ClassLabel'};
    noiseless_dt_temp_table = [filename_table noiseless_dt_class_label_table];
    noiseless_dt_testing_results = [noiseless_dt_testing_results; noiseless_dt_temp_table];
    
    %Noiseless Data KNN
    noiseless_knn_class_label = predict(noiseless_data_KNN_model, noiseless_data_table_transpose);
    noiseless_knn_class_label_table = cell2table({noiseless_knn_class_label});
    noiseless_knn_class_label_table.Properties.VariableNames = {'ClassLabel'};
    noiseless_knn_temp_table = [filename_table noiseless_knn_class_label_table];
    noiseless_knn_testing_results = [noiseless_knn_testing_results; noiseless_knn_temp_table];
    
    
    
    %DWT Data SVM
    dwt_svm_class_label = predict(dwt_data_SVM_model, dwt_data_table_transpose);
    dwt_svm_class_label_table = cell2table({dwt_svm_class_label});
    dwt_svm_class_label_table.Properties.VariableNames = {'ClassLabel'};
    dwt_svm_temp_table = [filename_table dwt_svm_class_label_table];
    dwt_svm_testing_results = [dwt_svm_testing_results; dwt_svm_temp_table];
    
    %DWT Data DT
    dwt_dt_class_label = predict(dwt_data_DT_model, dwt_data_table_transpose);
    dwt_dt_class_label_table = cell2table({dwt_dt_class_label});
    dwt_dt_class_label_table.Properties.VariableNames = {'ClassLabel'};
    dwt_dt_temp_table = [filename_table dwt_dt_class_label_table];
    dwt_dt_testing_results = [dwt_dt_testing_results; dwt_dt_temp_table];
    
    %DWT Data KNN
    dwt_knn_class_label = predict(dwt_data_KNN_model, dwt_data_table_transpose);
    dwt_knn_class_label_table = cell2table({dwt_knn_class_label});
    dwt_knn_class_label_table.Properties.VariableNames = {'ClassLabel'};
    dwt_knn_temp_table = [filename_table dwt_knn_class_label_table];
    dwt_knn_testing_results = [dwt_knn_testing_results; dwt_knn_temp_table];
    
end
print()