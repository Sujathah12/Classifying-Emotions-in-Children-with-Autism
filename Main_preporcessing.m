clc
clear 
close all
warning off
addpath(genpath('.'));
path=('./Train/');
Data=dir(path);
Data(1:2)=[];
out=[path(3:end-1),'_out'];
mkdir(out)
M1=1;
for N1=1:length(Data)
Read_fol=[path,Data(N1).name '/'];
    Read_fol1=dir(Read_fol);
    Read_fol1(1:2)=[];
    aa=[out '/',Data(N1).name];
    mkdir(aa)
    for N2=1:100
          Get=[Read_fol,Read_fol1(N2).name];
          %% Image Resizing
          I=imread (Get);
          resi=imresize(I,[224 224]);
          image = double(resi);
          normalized_image = (image - min(image(:))) / (max(image(:)) - min(image(:)));
          imshow(normalized_image); pause(0.01)
%           imwrite(im2double(normalized_image),[aa '/',Read_fol1(N2).name]);
          %% Rotation 
          angle=20;
          J = imrotate(I,angle);
          resij=imresize(J,[224 224]);
          imagej = double(resij);
          % Apply Min-Max normalization to scale pixel values to the range [0, 1]
          normalized_imagej = (imagej - min(imagej(:))) / (max(imagej(:)) - min(imagej(:)));
          imshow(normalized_imagej); pause(0.01)
%           imwrite(im2double(normalized_imagej),[aa '/','Rotating',Read_fol1(N2).name]);
          %% Mirroring
          Ir = flipdim(I,2); 
          resif=imresize(Ir,[224 224]);
          imagef = double(resif);
          %% Normalization 
          % Apply Min-Max normalization to scale pixel values to the range [0, 1]
          normalized_imagef = (imagef - min(imagef(:))) / (max(imagef(:)) - min(imagef(:)));
          imshow(normalized_imagef); pause(0.01)
%           imwrite(im2double(normalized_imagef),[aa '/','Mirroring',Read_fol1(N2).name]);
    end
end