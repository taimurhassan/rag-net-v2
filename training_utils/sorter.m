clc
clear all
close all

load gTruth.mat

for i = 1:length(gTruth.DataSource.Source)
    
    cell = gTruth.DataSource.Source(i);
    fn = cell{1};

    img = imread(fn);
    
    [filepath,name,ext] = fileparts(fn);
    
    img = imresize(img,[576 768],'bilinear');
    
    if contains(name,'CNV')
        imwrite(img,['dataset\CNV\' name '.png'],'PNG');    
    elseif contains(name,'DME')
        imwrite(img,['dataset\DME\' name '.png'],'PNG');
    elseif contains(name,'NORMAL')
        imwrite(img,['dataset\NORMAL\' name '.png'],'PNG');
    elseif contains(name,'DRUSEN')
        imwrite(img,['dataset\DRUSEN\' name '.png'],'PNG');
    end
end