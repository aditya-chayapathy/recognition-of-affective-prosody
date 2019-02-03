%MC Project
training_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Classification\Training\";
testing_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Classification\Testing\";
output_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Output\Final\MovMedian\";

%Creating output directory if it doesn't exist
if ~exist(output_folder_path, 'dir')
    mkdir(char(output_folder_path));
end

modified_file_path = strcat(training_folder_path, "\**\*.mat");
dir_info = dir(char(modified_file_path));
training_data_table = cell2table({});
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
    
    raw_data_table = data_table(1:end, 1:1);
    mov_median_data_array = movmedian(table2array(raw_data_table), 200);
    
    data_table_transpose = array2table(mov_median_data_array.');
    training_data_table = [training_data_table; data_table_transpose];
    
    training_class_labels_table = [training_class_labels_table; cell2table({class_name})];
end

%Training classification models
SVM_model = fitcsvm(training_data_table, training_class_labels_table);
DT_model = fitctree(training_data_table, training_class_labels_table);
KNN_model = fitcknn(training_data_table, training_class_labels_table);


modified_file_path = strcat(testing_folder_path, "\**\*.mat");
dir_info = dir(char(modified_file_path));

SVM_testing_results = cell2table({});
DT_testing_results = cell2table({});
KNN_testing_results = cell2table({});

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
    raw_data_table = data_table(1:end, 1:1);
    table_array = table2array(raw_data_table);
    data_table_transpose = array2table(table_array.');
    filename_table = cell2table({file_name});
    filename_table.Properties.VariableNames = {'FileName'};
    
    %SVM Classification
    SVM_class_label = predict(SVM_model, data_table_transpose);
    SVM_class_label_table = cell2table({SVM_class_label});
    SVM_class_label_table.Properties.VariableNames = {'ClassLabel'};
    SVM_temp_table = [filename_table SVM_class_label_table];
    SVM_testing_results = [SVM_testing_results; SVM_temp_table];
    
    %Noisy Data DT
    DT_class_label = predict(DT_model, data_table_transpose);
    DT_class_label_table = cell2table({DT_class_label});
    DT_class_label_table.Properties.VariableNames = {'ClassLabel'};
    DT_temp_table = [filename_table DT_class_label_table];
    DT_testing_results = [DT_testing_results; DT_temp_table];
    
    %Noisy Data KNN
    KNN_class_label = predict(KNN_model, data_table_transpose);
    KNN_class_label_table = cell2table({KNN_class_label});
    KNN_class_label_table.Properties.VariableNames = {'ClassLabel'};
    KNN_temp_table = [filename_table KNN_class_label_table];
    KNN_testing_results = [KNN_testing_results; KNN_temp_table];
end
writetable(SVM_testing_results, output_folder_path + "SVM_Results.csv");
writetable(KNN_testing_results, output_folder_path + "KNN_Results.csv");
writetable(DT_testing_results, output_folder_path + "DT_Results.csv");