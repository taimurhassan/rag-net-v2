clc
clear all
close all

load svm.mat

pn = 'C:\tbme\testingDataset\segmentation_results\';
pn2 = 'C:\tbme\testingDataset\test_images\';
pn3 = 'C:\tbme\testingDataset\diagnosisResults\';

imagefiles = dir([pn '*.png']);

nfiles = length(imagefiles);    

for ii=1:1:nfiles
    fn = imagefiles(ii).name;
    
    im=imread([pn fn]);
    
    load([pn3 replace(fn,'png','mat')]);
    
    if decision == 0 % if the scan is classified as normal, don't grade it further
        %fprintf(['Image: ' fn ', Normal Scan\n'])
        continue;
    end
    
    ori = imread([pn2 fn]);
    
    [im5,mask] = getImage(im);
    
    [r,c,ch] = size(im5);
       
	rnfl = zeros(1,c);
    gcipl = zeros(1,c);
    gcc= zeros(1,c);
    cc = [];
    ee = [];
    ff = [];
    d = [];
    
    for i = 1:c
        a = logical(im5(:,i,1));
        b = logical(im5(:,i,2));

        p1=find(a~=0,1,'first');
        p2=find(a~=0,1,'last');
        
        if ~isempty(p1) && ~isempty(p2)
            rnfl(i) = abs(p2-p1);
            cc = p1;
            ee = p2;
        end
        
        p1=find(b~=0,1,'first');
        p2=find(b~=0,1,'last');
        if ~isempty(p1) && ~isempty(p2)
            gcipl(i) = abs(p2-p1);
            d = p2;
            ff=p1;
        end
        
        if isempty(cc) && ~isempty(p1)
            cc = p1;
        end
        
        if gcipl(i) == 0 && ~isempty(ee) && ~isempty(cc)
            gcc(i) = abs(cc-ee);
        elseif rnfl(i) == 0 && ~isempty(ee) && ~isempty(ff)
            gcc(i) = abs(ff-ee);
        elseif ~isempty(cc) && ~isempty(d)
            gcc(i) = abs(cc-d);
        end
    end
    
    f1 = mean(rnfl);
    f2 = mean(gcipl);
    f3 = mean(gcc);
    
    class = predict(svm,[f1, f2, f3]);
    
    if class == 0
        fprintf(['Image: ' fn ', Grade: Early Glaucoma\n'])
    else
        fprintf(['Image: ' fn ', Grade: Advanced Glaucoma\n'])
    end
end

function [im5,im6] = getImage(im)
    l1 = [241, 169, 37];
    l2 = [207, 248, 132];
    l3 = [183, 244, 155];
    l4 = [222, 181, 51];
    l5 = [244, 104, 161];
    l6 = [144, 71, 111];
    l7 = [128, 48, 71];

    [r,c,ch] = size(im);
    
    im5 = zeros(r,c,ch);
    im6 = zeros(r,c);
    
    for i = 1:r
        for j = 1:c
            if im(i,j,1) == l1(1) && im(i,j,2) == l1(2) && im(i,j,3) == l1(3)
                im5(i,j,1) = 255;
                im5(i,j,2) = 0;
                im5(i,j,3) = 0;
                im6(i,j) = 1;
            elseif im(i,j,1) == l2(1) && im(i,j,2) == l2(2) && im(i,j,3) == l2(3)
                im5(i,j,1) = 0;
                im5(i,j,2) = 255;
                im5(i,j,3) = 0;
                im6(i,j) = 1;
            elseif im(i,j,1) == l3(1) && im(i,j,2) == l3(2) && im(i,j,3) == l3(3)
                im5(i,j,1) = 0;
                im5(i,j,2) = 0;
                im5(i,j,3) = 0;
                im6(i,j) = 0;
            end
        end
    end
    
    im6 = imfill(bwareaopen(logical(im6),160),'holes');
    
    im5(:,:,1) = im5(:,:,1) .* double(im6);
    im5(:,:,2) = im5(:,:,2) .* double(im6);
    im5(:,:,3) = im5(:,:,3) .* double(im6);
end