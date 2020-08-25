clc
clear all
close all

pn = 'F:\OCT Dataset\Training\UNKNOWN\';

imagefiles = dir([pn '*.jpg']);

nfiles = length(imagefiles);    % Number of files found

for ii=1:1:nfiles

fn = imagefiles(ii).name;
img=imread([pn fn]);

for t = 0:2:10
    img = rescale(img);
    img = imrotate(img,t,'bilinear','crop');
%     img = fliplr(img);
    img = img(:,:,2);
    
    if(size(img,3) ~= 3)
        img = cat(3,img,img,img);
    end
    
    img = imresize(img,[224 224],'bilinear');
    imwrite(img,[pn 'Resized\' num2str(ii) '_' num2str(t) '.jpg'],'JPEG');
end

end