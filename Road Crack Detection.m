clc;
clear;
close all;
imds1 = imageDatastore('C:\Users\manav\Desktop\gray\positive\*.jpg');

noi= length(imds1.Files);
out_pos=ones(noi,1);
for k = 1 : 1
  % Get the input filename.  It already has the folder prepended so we don't need to worry about that.
  inputFileName = imds1.Files{k};
  rgbImage = imread(inputFileName);
  [rows, columns, numberOfColorChannels] = size(rgbImage);
  if numberOfColorChannels == 3
    % It's color so need to convert to gray scale.
    grayImage = rgb2gray(rgbImage);
    %figure,imshow(grayImage); title('GRAY IMAGE');
  end
    %PRE-PROCESSING
    %1.RESIZE
    rs=imresize(grayImage,[480 640]);
    figure,imshow(rs);title('RESIZED IMAGE');
    
    %2.SMOOTHING
    h = fspecial('gaussian',[7 7],1);
    f=imfilter(rs,h);
    %figure,imshow(f); title('FILTERED IMAGE');
    
    %3.THRESHOLDING
    t=graythresh(f);
    BW = imbinarize(f,t);
    %figure,imshow(BW); title('OTSU THRESHOLDING');
    
    %4.MORPHOLOGICAL FILTERING
    BW2 = bwmorph(BW,'clean');
    e=im2uint8(BW2);
    %figure,imshow(BW2); title('MORPHOOGICAL FILTERING');
   
    %5.ENTROPY FILTERING
    J = entropyfilt(e);
    %figure,imshow(J); title('ENTROPY FILTERING');
    
    %6.AGAIN THRESHOLDING
    T=graythresh(J);
    BW3 = imbinarize(J,T);
    %e=im2double(BW3);
    %figure,imshow(BW3); title('THRESHOLDING AFTER ENTROPY FILTERING');
    
    
    
    %EXTRACING FEATURES
    h=imhist(BW3);
    ft_image=fft2(BW3);
    h1= imhist(ft_image);
    L1=extractLBPFeatures(BW3);
    S1=extractHOGFeatures(BW3);
    
    fe1_pos(k,:)=h1;
    fe2_pos(k,:)=L1;
    fe3_pos(k,:)=S1;
end


imds2 = imageDatastore('C:\Users\manav\Desktop\gray\negative\*.jpg');

no= length(imds2.Files);
out_neg=zeros(no,1);
for k = 1 : no
  % Get the input filename.  It already has the folder prepended so we don't need to worry about that.
  inputFileName = imds2.Files{k};
  rgbImage = imread(inputFileName);
[rows, columns, numberOfColorChannels] = size(rgbImage);
 if numberOfColorChannels == 3
    % It's color so need to convert to gray scale.
    grayImage = rgb2gray(rgbImage);
 end
    rs=imresize(grayImage,[480 640]);
    %figure,imshow(rs);
    
    h = fspecial('gaussian',[7 7],1);
    f=imfilter(rs,h);
    
    t=graythresh(f);
    BW = imbinarize(f,t);
    
    BW2 = bwmorph(BW,'clean');
    e=im2uint8(BW2);

    J = entropyfilt(e);
    
    T=graythresh(J);
    BW3 = imbinarize(J,T);
    
    h=imhist(BW3);
    ft_image=fft2(BW3);
    h1= imhist(ft_image);
    L1=extractLBPFeatures(BW3);
    S1=extractHOGFeatures(BW3);
    
    fe1_neg(k,:)=h1;
    fe2_neg(k,:)=L1;
    fe3_neg(k,:)=S1;
end

fe1=[fe1_pos ; fe1_neg];
fe2=[fe2_pos ; fe2_neg];
fe3=[fe3_pos ; fe3_neg];
feF=[fe1  fe2  fe3];
output=[out_pos;out_neg];

 
imds3 = imageDatastore('C:\Users\manav\Desktop\test\pos\*.jpg');

not= length(imds3.Files);
out_posT=ones(not,1);
for k = 1 : not
  % Get the input filename.  It already has the folder prepended so we don't need to worry about that.
  inputFileName = imds3.Files{k};
  rgbImage = imread(inputFileName);
[rows, columns, numberOfColorChannels] = size(rgbImage);
 if numberOfColorChannels == 3
    % It's color so need to convert to gray scale.
    grayImage = rgb2gray(rgbImage);
 end
    rs=imresize(grayImage,[480 640]);
    
    h = fspecial('gaussian',[7 7],1);
    f=imfilter(rs,h);
   
    t=graythresh(f);
    BW = imbinarize(f,t);
    
    BW2 = bwmorph(BW,'clean');
    e=im2uint8(BW2);
  
    J = entropyfilt(e);
    
    T=graythresh(J);
    BW3 = imbinarize(J,T);
    
    h=imhist(BW3);
    ft_image=fft2(BW3);
    h1= imhist(ft_image);
    L1=extractLBPFeatures(BW3);
    S1=extractHOGFeatures(BW3);
    
    fe1_posT(k,:)=h1;
    fe2_posT(k,:)=L1;
    fe3_posT(k,:)=S1;
end

 imds4 = imageDatastore('C:\Users\manav\Desktop\test\neg\*.jpg');

noT= length(imds4.Files);
out_negT=zeros(noT,1);
for k = 1 : noT
  % Get the input filename.  It already has the folder prepended so we don't need to worry about that.
  inputFileName = imds4.Files{k};
  rgbImage = imread(inputFileName);
[rows, columns, numberOfColorChannels] = size(rgbImage);
 if numberOfColorChannels == 3
    % It's color so need to convert to gray scale.
    grayImage = rgb2gray(rgbImage);
 end
    rs=imresize(grayImage,[480 640]);
  
    h = fspecial('gaussian',[7 7],1);
    f=imfilter(rs,h);
  
    t=graythresh(f);
    BW = imbinarize(f,t);
    
    BW2 = bwmorph(BW,'clean');
    e=im2uint8(BW2);
    
    J = entropyfilt(e);
    
    T=graythresh(J);
    BW3 = imbinarize(J,T);
    
    h=imhist(BW3);
    ft_image=fft2(BW3);
    h1= imhist(ft_image);
    L1=extractLBPFeatures(BW3);
    S1=extractHOGFeatures(BW3);
    
    fe1_negT(k,:)=h1;
    fe2_negT(k,:)=L1;
    fe3_negT(k,:)=S1;
end

fe1T=[fe1_posT ; fe1_negT];
fe2T=[fe2_posT ; fe2_negT];
fe3T=[fe3_posT ; fe3_negT];
fef=[fe1T  fe2T  fe3T];
outputT=[out_posT;out_negT];
%outputT=imresize(outputT,[2*length(out_posT),1]);

 model=fitctree(feF,output);
 z=predict(model,fef);
 conf=confusionmat(outputT,z);
 q=bcd(conf);

 
 
 

