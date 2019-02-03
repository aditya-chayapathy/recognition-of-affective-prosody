%MC Project
training_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Classification\Training\";
testing_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Classification\Testing\";
output_folder_path = "D:\Kevin Thomas\ASU\3rd Semester\MC\Project\MCProject\Data\Output\DWT\";

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
    
    raw_data_table = data_table(500:2250, 1:1);
    %mov_median_data_array = movmedian(table2array(raw_data_table), 200);
    dwt_features_data_array = dwt(table2array(raw_data_table), 'sym4');
    
    figure = plot(dwt_features_data_array);
    title(file_name);
    output_file_path = char(strcat(output_folder_path, file_name, ".png"));
    saveas(figure, output_file_path)
end