clc
clear all
close all

pn = 'D:\TBME\AFIO\afio\Training\normal\';

imagefiles = dir([pn '*.png']);

nfiles = length(imagefiles);    % Number of files found

for ii=1:1:nfiles

fn = imagefiles(ii).name;
img=imread([pn fn]);

for t = -5:1:5
    a = fliplr(img);
%     img = rescale(img);
    img = imrotate(img,t,'bilinear','crop');
	c = imrotate(a,t,'bilinear','crop');
%     img = fliplr(img);
%     img = img(:,:,2);
    
%     if(size(img,3) ~= 3)
%         img = cat(3,img,img,img);
%     end
    
%     img = imresize(img,[224 224],'bilinear');
    imwrite(a,[pn 'Resized\' num2str(ii) '_' num2str(t) '1.jpg'],'JPEG');
    imwrite(c,[pn 'Resized\' num2str(ii) '_' num2str(t) '2.jpg'],'JPEG');
    imwrite(img,[pn 'Resized\' num2str(ii) '_' num2str(t) '3.jpg'],'JPEG');
end

end