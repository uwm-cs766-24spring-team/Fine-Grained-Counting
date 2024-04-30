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
dm_towards = fullfile(dm_path_prefix, 'towards/');
dm_away = fullfile(dm_path_prefix, 'away/');

fprintf('Path: %s\n', gt_path_prefix);

gt_files = dir([gt_path_prefix '*.mat']);
num_files = length(gt_files);
img_suffix = '.jpg';

if (dataset_name == 'Towards_vs_Away')
    img_suffix = '.png';
end

fprintf('Number of files: %d\n', num_files);

mkdir(dm_towards)
mkdir(dm_away)

for i = 1:num_files
    if (mod(i,10)==0)
        fprintf(1,'Processing %3d/%d files\n', i, num_files);
    end
    gt_file_name = gt_files(i).name;
    filename = gt_file_name(1:end-4);
    
    load(strcat(gt_path_prefix, gt_file_name)) ;
    load(strcat(current_path, '/dataset/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_1.mat')) ;
    
    input_img_name = strcat(path, filename, img_suffix);
    im = imread(input_img_name);
    [~, ~, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end
    [h, w, c] = size(im);
    towards_im_density = get_density_map_gaussian(im, towards, 15, 4.0);
    csvwrite([dm_towards, filename '.csv'], towards_im_density);
    
    away_im_density = get_density_map_gaussian(im, away, 15, 4.0);
    csvwrite([dm_away, filename '.csv'], away_im_density);
end

