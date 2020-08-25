clc
clear all
close all


pn = 'C:\jbhi_code\results\preprocessing\';

imagefiles = dir([pn '*.png']);

nfiles = length(imagefiles);    % Number of files found

for ii = 1:nfiles
    fn = imagefiles(ii).name;
    img=imread([pn fn]);    
    
    if(size(img,3) == 3)
        img = img(:,:,2);
        img(img>200) = 0;
    end

    img(1:50,:) = 0;

    img(end-50:end,:) = 0;

    img = imresize(img,[480 1280],'bilinear');

    img = medfilt2(img);

    img(img<30) = 0;

    oimg = img;
    
    [s1, s2, s3] = structureTensor(img,10,1);

    s1 = mat2gray(s1);
    s2 = mat2gray(s2);
    s3 = mat2gray(s3);

    [~,e1,~] = svd(s1);
    [~,e2,~] = svd(s2);
    [~,e3,~] = svd(s3);
        
    e1 = max(max(e1));
    e2 = max(max(e2));
    e3 = max(max(e3));
    
    ec = median([e1,e2,e3]);

    if e1 == ec
        s3 = s1;
    elseif e2 == ec
        s3 = s2;
    end
    
    s3=3*s3;

    [r,c] = size(s3);

    [s3, mask] = hysteresis3d(s3,0.02,0.1,4);
   
    s3 = imfill(s3,'holes');    
    s3 = bwareaopen(s3,3000);

    sTImg = edge(s3,'canny');
    sTImg = bwmorph(sTImg,'skel');

    sTImg(end-100:end,:) = 0;

    skImg = zeros(480,1280);
    first = zeros(1,length(sTImg(1,:)));
    second = zeros(1,length(sTImg(1,:)));

    img = imcrop(img,[10 0 length(img(1,:)) length(img(:,1))]);

    lastPointX=0;
    lastPointY=0;

    img = sTImg;
    [r,c] = size(img);

    first = NaN(1,c);
    last = NaN(1,c);
    lastPointX = 0;
    lastPointY = 0;

    for i = 1:c

            p1=find(img(:,i)~=0,1,'first');
            p2=find(img(:,i)~=0,1,'last');

            if(~isempty(p1) && ~isempty(p2))
                img(p1,i) = 0;
                img(p2,i) = 0;
                p1 = p1 + 10;
                p2 = p2 - 15;

                first(1,i) = p1;
                last(1,i) = p2;

                if(i - 1 == 1)
                    lastPointX = first(1,i);
                    lastPointY = last(1,i);
                end

                if(i - 1 > 1)
                    dist1 = sqrt((i-i-1).^2 + (first(1,i)-lastPointX).^2);
                    dist2 = sqrt((i-i-1).^2 + (last(1,i)-lastPointY).^2);           

                    if(dist1 < 2000)
                        lastPointX = first(1,i);
                    else
                        first(1,i) = NaN;
                    end

                    if(dist2 < 2000)
                        lastPointY = last(1,i);
                    else
                        last(1,i) = NaN;
                    end
                end

            else
                if(i-1>0)
                    first(1,i) = first(1,i-1);
                    last(1,i) = last(1,i-1);
                end
            end        
    end
    
    first = abs(first);
    last = abs(last);

    mask = zeros(size(oimg(:,:,1)));
    for i = 1:c
        if(~isnan(first(i)) && ~isnan(last(i)) && last(i) > 1 && last(i) < r)
            mask((first(i)):(last(i)),i) = 1;
        end
    end

    img = mat2gray(mask.*double(oimg));

    img = imresize(img,[576 768],'bilinear');
    
    % create folder with name "Preprocessed" in the preprocessing directory
    % before saving the preprocessed image 
    imwrite(img,[pn 'Preprocessed\' fn],'PNG');
end

%%
function [Sxx, Sxy, Syy] = structureTensor(I,si,so)
I = double(I);
[m n] = size(I);
 
Sxx = NaN(m,n);
Sxy = NaN(m,n);
Syy = NaN(m,n);
 
x  = -2*si:2*si;
g  = exp(-0.5*(x/si).^2);
g  = g/sum(g);
gd = -x.*g/(si^2); 
 
Ix = conv2( conv2(I,gd,'same'),g','same' );
Iy = conv2( conv2(I,gd','same'),g,'same' );
 
Ixx = Ix.^2;
Ixy = Ix.*Iy;
Iyy = Iy.^2;
 
x  = -2*so:2*so;
g  = exp(-0.5*(x/so).^2);
Sxx = conv2( conv2(Ixx,g,'same'),g','same' ); 
Sxy = conv2( conv2(Ixy,g,'same'),g','same' );
Syy = conv2( conv2(Iyy,g,'same'),g','same' );

end

%% Hysteresis3D

function [tri,hys]=hysteresis3d(img,t1,t2,conn)
if nargin<3
    disp('function needs at least 3 inputs')
    return;
elseif nargin==3
    disp('inputs=3')
    if numel(size(img))==2;
        disp('img=2D')
        disp('conn set at 4 connectivies (number of neighbors)')
        conn=8;
    end
    if numel(size(img))==3; 
        disp('img=3D')
        disp('conn set at 6 connectivies (number of neighbors)')
        conn=6;
    end
end

if t1>t2    
	tmp=t1;
	t1=t2; 
	t2=tmp;
end
minv=min(img(:));               
maxv=max(img(:));                
t1v=t1*(maxv-minv)+minv;
t2v=t2*(maxv-minv)+minv;

tri=zeros(size(img));
tri(img>=t1v)=1;
tri(img>=t2v)=2;

abovet1=img>t1v;                                     
seed_indices=sub2ind(size(abovet1),find(img>t2v));   
hys=imfill(~abovet1,seed_indices,conn);              
hys=hys & abovet1;
end
