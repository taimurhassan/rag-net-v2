clc
clear all
close all

workingDir = 'testingDataset\test_images\';

pn = 'testingDataset\segmentation_results\';

imageNames = dir(fullfile(pn,'*.png'));
imageNames = {imageNames.name}';

outputVideo = VideoWriter(fullfile(workingDir,'video'));
outputVideo.FrameRate = 10;
open(outputVideo)

for ii = 1:length(imageNames)
   im2 = imread(fullfile(pn,imageNames{ii}));
   img = imread([workingDir imageNames{ii}]);

   im = montage({img, im2})
   
   im = imresize(im.CData,[854, 2077],'bilinear');
   writeVideo(outputVideo,im)
end

close(outputVideo)
