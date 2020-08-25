clc
clear all
close all

workingDir = 'C:\oct_diagnosis\jbhi_code\code\testingDataset\test_images\';

pn = 'C:\oct_diagnosis\jbhi_code\code\testingDataset\segmentation_results\';

imageNames = dir(fullfile(pn,'*.png'));
imageNames = {imageNames.name}';

outputVideo = VideoWriter(fullfile(workingDir,'video'));
outputVideo.FrameRate = 10;
open(outputVideo)

% bg = [20 215 197];
% one = [207 248 132];
% two = [144 71 111];
% three = [183 244 155];

bg = [20 215 197];
one = [207 248 132];
two = [144 71 111];
three = [183 244 155];
four = [128 48 71];

for ii = 1:length(imageNames)
%    img = imread(fullfile(workingDir,imageNames{ii}));
%    imgo = img;
   im2 = imread(fullfile(pn,imageNames{ii}));
   img = imread([workingDir imageNames{ii}]);
%    
%    [r,c,ch] = size(im2);
%    mask = zeros(r,c);
%    for i = 1:r
%        for j = 1:c
%            if im2(i,j,1) == one(1) && im2(i,j,2) == one(2) && im2(i,j,3) == one(3)
% %                img(i,j,1) = 0;
%                img(i,j,2) = 0;
%                img(i,j,3) = 0;
%            elseif im2(i,j,1) == two(1) && im2(i,j,2) == two(2) && im2(i,j,3) == two(3)
%                img(i,j,1) = 0;
% %                mask(i,j) = 1;
%                img(i,j,3) = 0;               
%            elseif im2(i,j,1) == three(1) && im2(i,j,2) == three(2) && im2(i,j,3) == three(3)
%                img(i,j,1) = 0;
%                img(i,j,2) = 0;
% %                img(i,j,3) = 0;              
%            elseif im2(i,j,1) == four(1) && im2(i,j,2) == four(2) && im2(i,j,3) == four(3)
% %                img(i,j,1) = 0;
% %                img(i,j,2) = 0;
%                img(i,j,3) = 0;
%            end
%        end
%    end
%    
% %    mask = imfill(mask,'holes');
% %    mask = bwareaopen(logical(mask),400);
% %    
% %    b = img(:,:,1);
% %    b(mask == 1) = 0;
% %    img(:,:,1) = b;
% %    b = img(:,:,3);
% %    b(mask == 1) = 0;
% %    img(:,:,3) = b;
%    
% %     img(:,:,2) = 255 * uint8(mask);
   
   im = montage({img, im2})
   
   im = imresize(im.CData,[854, 2077],'bilinear');
   writeVideo(outputVideo,im)
end

close(outputVideo)
