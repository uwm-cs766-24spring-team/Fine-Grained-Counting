%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create grount truth density map for test set%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear all;
cd(fileparts(mfilename('fullpath')));
cd ../../../

dataset_name = 'Towards_vs_Away';
current_path = pwd;

path_prefix = ['dataset/Fine-Grained-Counting-Dataset/' dataset_name '/'];
path_prefix_abs = fullfile(current_path, path_prefix);
path = fullfile(path_prefix_abs, 'images/');
gt_path_prefix = fullfile(path_prefix_abs, 'ground_truth/');
dm_path_prefix = fullfile(path_prefix_abs, 'density_maps/');

fprintf('Path: %s\n', gt_path_prefix);

gt_files = dir([gt_path_prefix '*.mat']);
num_files = length(gt_files);
img_suffix = '.jpg';

if (dataset_name == 'Towards_vs_Away')
    img_suffix = '.png';
end

fprintf('Number of files: %d\n', num_files);

mkdir(dm_path_prefix)

for i = 1:num_files
    if (mod(i,10)==0)
        fprintf(1,'Processing %3d/%d files\n', i, num_files);
    end
    gt_file_name = gt_files(i).name;
    filename = gt_file_name(1:end-4);
    
    load(strcat(gt_path_prefix, gt_file_name)) ;
    
    input_img_name = strcat(path, filename,img_suffix);
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end
    annPoints = towards;
    [h, w, c] = size(im);
    im_density = get_density_map_gaussian(im,annPoints, 4,15);
    csvwrite([gt_path_csv ,filename '.csv'], im_density);
end

