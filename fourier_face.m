%store all 400 images
imds = imageDatastore('ATT_Face_Database/all','IncludeSubfolders', true);
collage = imread('collage.jpg');
%calculate fourier transform for 200 training images
training_set_freq = [];
count = 0; %for recongition rate
total = 0;
for i = 0:39
    for j = 6:10
       training = readimage(imds,i*10+j);
       training_freq = ff_space(training);
       training_set_freq = cat(3,training_set_freq,training_freq);
    end
end
%find index of maximum variance frequency
variance = var(training_set_freq,0,3);
ind_var = zeros(22,2);
for i = 1:22
    [m,ind] = max(variance(:));
    [ind_var(i,1),ind_var(i,2)]=ind2sub(size(variance),ind);
    variance(ind_var(i,1),ind_var(i,2)) = 0+0*i;
end
%get frequency from index
var_set_freq = zeros(22,200);
for i = 1:22
    for j = 1:200
        var_set_freq(i,j) = training_set_freq(ind_var(i,1),ind_var(i,2),j);
    end
end
%get 22 real frequency and 8 imaginary frequency
real_freq = real(var_set_freq);
imag_freq = imag(var_set_freq(1:8,:));
training_freq = cat(1,real_freq,imag_freq);
%training_freq = abs(var_set_freq);     %magnitude test

%create collage
%{
image = readimage(imds,1);
collage = zeros(size(image,1)*5,size(image,2)*8);
for y = 1:5
    for x = 1:8
        image = readimage(imds,(((y-1)*8+x)*10-9));
        for h = 1:size(image,1)
            for w = 1:size(image,2)
                collage((y-1)*size(image,1)+h,(x-1)*size(image,2)+w) = image(h,w);
            end
        end
    end
end
imwrite(uint8(collage),'collage.jpg','jpg');
input('Hit any key to continue.');
%}
%get the model image
%a=1;
while 1 %a<401
total = total+1;
a = randi(400);
image = readimage(imds,a);
img_size = size(image);
freq = ff_space(image);
%for rotation test
%{
%Padding on image so the size will be power of 2
image = padarray(image,[double((2^(ceil(log2(img_size(1))))-img_size(1))/2) double((2^(ceil(log2(img_size(2))))-img_size(2))/2)],'replicate');
image = double(image)/255;
image = imrotate(image,10);    
freq = fft2(image);
%}
%get frequency from index
image_var_set_freq = zeros(22,1);
for i = 1:22
    image_var_set_freq(i) = freq(ind_var(i,1),ind_var(i,2));
end
%get 22 real frequency and 8 imaginary frequency
image_real_freq = real(image_var_set_freq);
image_imag_freq = imag(image_var_set_freq(1:8,:));
image_freq = cat(1, image_real_freq,image_imag_freq);
%image_freq = abs(image_var_set_freq);  %magnitude test

%calculate the euclidean distance
norm_i = zeros(40,5);
for i = 1:40
    for j = 1:5
        norm_i(i,j) = norm(training_freq(:,(i-1)*5+j)-image_freq(:));
    end
end

%calculate the mean of euclidean distance
mean_norm = mean(norm_i,2);
%find the number which return the minimum distance
[v,num] = min(mean_norm);
result = readimage(imds,(num-1)*10+1);

figure('Position',[100, 100, 1200, 800]);
subplot(5,10,21);
imshow(image);
title('input image','FontSize',20);
subplot(5,10,[3,50]);
imshow(collage);
hold on;
if ceil(a/10) == num
    count = count+1;
    title('Match','FontSize',30,'Color','g');
else
    title('NO match','FontSize',30,'Color','r');
end
if rem(num,8) == 0
    r = 7;
else
    r = rem(num,8)-1;
end
rectangle('Position',[r*img_size(2) (ceil(num/8)-1)*img_size(1) img_size(2) img_size(1)],'LineWidth',3,'EdgeColor','r');
hold off;
disp('recognition rate:');
disp(count/total);
input('Hit any key to continue.');
close;
%a=a+1;
end
disp('recognition rate:');
disp(count/400);
function freq = ff_space(image)
    img_size = size(image);
    %Padding on image so the size will be power of 2
    image = padarray(image,[double((2^(ceil(log2(img_size(1))))-img_size(1))/2) double((2^(ceil(log2(img_size(2))))-img_size(2))/2)],'replicate');
    image = double(image)/255;
    freq = fft2(image);
end
