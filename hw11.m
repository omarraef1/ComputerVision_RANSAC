function hw11()

close all;
%
%%
%%% PART A %%%
data1 = load('line_data_2.txt');
whos('data1')
%data1(:,1)
figure(1),
scatter(data1(:,1),data1(:,2));
hold on
data2 = [data1(:,1) data1(:,2) ones(size(data1(:,1)))];
data = data1;

K=100;
sigma=1;
pretotal=0;
k=1;
while pretotal < size(data,1)*2/3 &&  k<K
    SampIndex=floor(1+(size(data,1)-1)*rand(2,1));
    
    samp1=data(SampIndex(1),:);
    samp2=data(SampIndex(2),:);
    
    line=tls([samp1;samp2]);
    mask=abs(line*[data ones(size(data,1),1)]');
    total=sum(mask<sigma);
    
    if total>pretotal
        pretotal=total;
        bestline=line;
    end  
    k=k+1;
end

mask=abs(bestline*[data ones(size(data,1),1)]')<sigma;    
hold on

finalMat = [0 0];
for i=1:length(mask)
    if mask(i)
        if i==1
            finalMat = [data(i,1) data(i,2)];
        else
            finalMat = [finalMat; data(i,1) data(i,2)];
        end
    end
end
plot(finalMat(:,1), finalMat(:,2), 'r.')
hold on
coeffs = polyfit(finalMat(:,1), finalMat(:,2),1);
fittedX = linspace(min(finalMat(:,1)), max(finalMat(:,1)), 100200);
fittedY = polyval(coeffs, fittedX);
%hold on
plot(fittedX, fittedY, 'r-', 'LineWidth', 3)
%{
for i=1:length(mask)
    if mask(i)
        plot(data(i,1),data(i,2),'r.');
    end
end
%}
%{
ff = polyfit(data(:,1), data(:,2),1);
fff = polyval(ff, data(:,1));
plot(data(:,1), fff, '--r')
%}
hold off
hold off

%% part B1
%Homography
%4 pairs (repeat 9 more)
p4_1 = randi([0 1], 1, 2);
p4_2 = randi([0 1], 1, 2);
p4_3 = randi([0 1], 1, 2);
p4_4 = randi([0 1], 1, 2);
p4_5 = randi([0 1], 1, 2);
p4_6 = randi([0 1], 1, 2);
p4_7 = randi([0 1], 1, 2);
p4_8 = randi([0 1], 1, 2);

ptset1 = [p4_1 1; p4_2 1; p4_3 1; p4_4 1];
ptset1t = ptset1';
whos('ptset1t')
whos('ptset1')
ptset2 = [p4_5 1; p4_6 1; p4_7 1; p4_8 1];
ptset2t = ptset2';

rmsavg4 = 0;
for i = 1:10
[H4, T41, T42] = homography(ptset1t, ptset2t);


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)

%rms??
diff = T41 - T42;
squared = diff.^2;
summed = sum(squared, 'all');
division = summed/15;
rms = sqrt(division);
rms

rmsavg4 = rmsavg4 + rms;
end
rmsavg4
rmsavg4fin = rmsavg4/10;
rmsavg4fin
%repeat 9 times and average %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%5 pairs (repeat 9 more)
p5_1 = randi([0 1], 1, 2);
p5_2 = randi([0 1], 1, 2);
p5_3 = randi([0 1], 1, 2);
p5_4 = randi([0 1], 1, 2);
p5_5 = randi([0 1], 1, 2);
p5_6 = randi([0 1], 1, 2);
p5_7 = randi([0 1], 1, 2);
p5_8 = randi([0 1], 1, 2);
p5_9 = randi([0 1], 1, 2);
p5_10 = randi([0 1], 1, 2);

ptset1 = [p5_1 1; p5_2 1; p5_3 1; p5_4 1; p5_5 1];
ptset1t = ptset1';
whos('ptset1t')
whos('ptset1')
ptset2 = [p5_6 1; p5_7 1; p5_8 1; p5_9 1; p5_10 1];
ptset2t = ptset2';

rmsavg5 = 0;
for i = 1:10
[H4, T41, T42] = homography(ptset1t, ptset2t);

%rms??
diff = T41 - T42;
squared = diff.^2;
summed = sum(squared, 'all');
division = summed/15;
rms = sqrt(division);
rms

rmsavg5 = rmsavg5 + rms;
end
rmsavg5
rmsavg5fin = rmsavg5/10;
rmsavg5fin
%repeat 9 times and average %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%6pairs (repeat 9 more)
p6_1 = randi([0 1], 1, 2);
p6_2 = randi([0 1], 1, 2);
p6_3 = randi([0 1], 1, 2);
p6_4 = randi([0 1], 1, 2);
p6_5 = randi([0 1], 1, 2);
p6_6 = randi([0 1], 1, 2);
p6_7 = randi([0 1], 1, 2);
p6_8 = randi([0 1], 1, 2);
p6_9 = randi([0 1], 1, 2);
p6_10 = randi([0 1], 1, 2);
p6_11 = randi([0 1], 1, 2);
p6_12 = randi([0 1], 1, 2);

ptset1 = [p6_1 1; p6_2 1; p6_3 1; p6_4 1; p6_5 1; p6_6 1];
ptset1t = ptset1';
whos('ptset1t')
whos('ptset1')
ptset2 = [p6_7 1; p6_8 1; p6_9 1; p6_10 1; p6_11 1; p6_12 1];
ptset2t = ptset2';

rmsavg6 = 0;
for i = 1:10
[H4, T41, T42] = homography(ptset1t, ptset2t);

%rms??
diff = T41 - T42;
squared = diff.^2;
summed = sum(squared, 'all');
division = summed/15;
rms = sqrt(division);
rms

rmsavg6 = rmsavg6 + rms;
end
rmsavg6
rmsavg6fin = rmsavg6/10;
rmsavg6fin
%repeat 9 times and average %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% PART B2
%slides and frames choose 8 points and run DLT
%then rms the transformations 
% (TRIAL WITH X AND Y) (Try orientations)

% SLIDE FRAME 1
slide1 = imread('slide1.tiff');
slide1 = slide1(:,:,1:3);
frame1 = imread('frame1.jpg');
ptset1 = [3 3 1; 332 24 1; 108 16 1; 256 85 1; 217 221 1; 202 157 1; 183 151 1; 316 114 1];
ptset1t = ptset1';
whos('ptset1t')
whos('ptset1')
ptset2 = [148 15 1; 324 30 1; 204 22 1; 277 54 1; 262 123 1; 254 92 1; 245 86 1; 315 71 1];
ptset2t = ptset2';
[Hsf1, T41, T42] = homography(ptset1t, ptset2t);
Hsf1
%rms??
diff = T41 - T42;
squared = diff.^2;
summed = sum(squared, 'all');
division = summed/15;
rms = sqrt(division);
rms

figure(2), imshow(slide1);
hold on
plot(ptset1(:,1), ptset1(:,2), 'r*')
hold off
figure(3), imshow(frame1)
datacursormode on
hold on
ptset2(:,1)
ptset2(:,2)
%plot(148, 15, 'r*')
hold on
plot(ptset2(:,1), ptset2(:,2), 'r*')
hold off

% SLIDE FRAME 2
slide2 = imread('slide2.tiff');
frame2 = imread('frame2.jpg');
ptset1_2 = [166 227 1; 134 221 1; 121 178 1; 163 179 1; 204 179 1; 242 176 1; 252 183 1; 277 145 1];
ptset1t_2 = ptset1_2';
whos('ptset1t')
whos('ptset1')
ptset2_2 = [162 145 1; 143 139 1; 134 116 1; 161 116 1; 186 116 1; 207 116 1; 214 121 1; 227 95 1];
ptset2t_2 = ptset2_2';
[Hsf2, T41, T42] = homography(ptset1t_2, ptset2t_2);
Hsf2
%rms??
diff = T41 - T42;
squared = diff.^2;
summed = sum(squared, 'all');
division = summed/15;
rms = sqrt(division);
rms

figure(4), imshow(slide2);
hold on
plot(ptset1_2(:,1), ptset1_2(:,2), 'r*')
hold off
figure(5), imshow(frame2)
datacursormode on
hold on
ptset2_2(:,1)
ptset2_2(:,2)
%plot(148, 15, 'r*')
hold on
plot(ptset2_2(:,1), ptset2_2(:,2), 'r*')
hold off

% SLIDE FRAME 3
slide3 = imread('slide3.tiff');
frame3 = imread('frame3.jpg');
ptset1_3 = [101 114 1; 81 115 1; 35 107 1; 113 101 1; 120 103 1; 181 101 1; 191 101 1; 211 101 1];
ptset1t_3 = ptset1_3';
whos('ptset1t')
whos('ptset1')
ptset2_3 = [119 58 1; 98 60 1; 49 51 1; 133 46 1; 140 48 1; 203 47 1; 215 47 1; 242 48 1];
ptset2t_3 = ptset2_3';
[Hsf3, T41, T42] = homography(ptset1t_3, ptset2t_3);
Hsf3
%rms??
diff = T41 - T42;
squared = diff.^2;
summed = sum(squared, 'all');
division = summed/15;
rms = sqrt(division);
rms

figure(6), imshow(slide3);
hold on
plot(ptset1_3(:,1), ptset1_3(:,2), 'r*')
hold off
figure(7), imshow(frame3)
datacursormode on
hold on
ptset2_3(:,1)
ptset2_3(:,2)
%plot(148, 15, 'r*')
hold on
plot(ptset2_3(:,1), ptset2_3(:,2), 'r*')
hold off

% PART B3
%take 4 of the 8 points and run homography on them
%use the computed homography to map points onto the video frame
% X' = H*X
% 
%slide frame 1
whos('ptset1')
whos('Hsf1')
nupts = [101 114 1; 35 107 1; 113 101 1; 211 101 1];
nupts2 = [119 58 1; 49 51 1; 133 46 1; 242 48 1];
nuptst = nupts';
nupts2t = nupts2';
[Hsfc1, T4c1, T4c2] = homography(nuptst, nupts2t);
whos('nupts')
reproj1 = Hsfc1*ptset1';
reproj1
reproj1(1,:)
reproj1F = [reproj1(1,:)./reproj1(3,:); reproj1(2,:)./reproj1(3,:)];
reproj1F
ptset2
figure(8), imshow(slide1)
hold on
plot(ptset1(:,1), ptset1(:,2), 'r*')
hold on
plot(nupts(:,1), nupts(:,2), 'y*')
hold off
figure(9), imshow(frame1)
datacursormode on
hold on
plot(reproj1F(1,:), reproj1F(2,:), 'r*')
hold on
plot(nupts2(:,1), nupts2(:,2), 'y*')
hold off

%slide frame 2
whos('ptset1')
whos('Hsf1')
nupts = [166 227 1; 134 221 1; 204 179 1; 242 176 1];
nupts2 = [162 145 1; 143 139 1; 186 116 1; 207 116 1];
nuptst = nupts';
nupts2t = nupts2';
[Hsfc2, T4c1, T4c2] = homography(nuptst, nupts2t);
whos('nupts')
reproj1 = Hsfc2*ptset1_2';
reproj1
reproj1(1,:)
reproj1F = [reproj1(1,:)./reproj1(3,:); reproj1(2,:)./reproj1(3,:)];
reproj1F
ptset2
figure(8), imshow(slide2)
hold on
plot(ptset1_2(:,1), ptset1_2(:,2), 'r*')
hold on
plot(nupts(:,1), nupts(:,2), 'y*')
hold off
figure(9), imshow(frame2)
datacursormode on
hold on
plot(reproj1F(1,:), reproj1F(2,:), 'r*')
hold on
plot(nupts2(:,1), nupts2(:,2), 'y*')
hold off


ptset1_3 = [101 114 1; 81 115 1; 35 107 1; 113 101 1; 120 103 1; 181 101 1; 191 101 1; 211 101 1];
ptset1t_3 = ptset1_3';
whos('ptset1t')
whos('ptset1')
ptset2_3 = [119 58 1; 98 60 1; 49 51 1; 133 46 1; 140 48 1; 203 47 1; 215 47 1; 242 48 1];
ptset2t_3 = ptset2_3';

%slide frame 3
whos('ptset1')
whos('Hsf1')
nupts = [101 114 1; 81 115 1; 120 103 1; 181 101 1];
nupts2 = [119 58 1; 98 60 1; 140 48 1; 203 47 1];
nuptst = nupts';
nupts2t = nupts2';
[Hsfc3, T4c1, T4c2] = homography(nuptst, nupts2t);
whos('nupts')
reproj1 = Hsfc3*ptset1_3';
reproj1
reproj1(1,:)
reproj1F = [reproj1(1,:)./reproj1(3,:); reproj1(2,:)./reproj1(3,:)];
reproj1F
ptset2
figure(8), imshow(slide3)
hold on
plot(ptset1_3(:,1), ptset1_3(:,2), 'r*')
hold on
plot(nupts(:,1), nupts(:,2), 'y*')
hold off
figure(9), imshow(frame3)
datacursormode on
hold on
plot(reproj1F(1,:), reproj1F(2,:), 'r*')
hold on
plot(nupts2(:,1), nupts2(:,2), 'y*')
hold off


%% PART C

end

%-----------------------------------------------------------------------
% Function to check argument values and set defaults

function [x1, x2] = layout(arg);
    
    if length(arg) == 2
	x1 = arg{1};
	x2 = arg{2};
	if ~all(size(x1)==size(x2))
	    error('x1 and x2 must have the same size');
	elseif size(x1,1) ~= 3
	    error('x1 and x2 must be 3xN');
	end
	
    elseif length(arg) == 1
	if size(arg{1},1) ~= 6
	    error('Single argument x must be 6xN');
	else
	    x1 = arg{1}(1:3,:);
	    x2 = arg{1}(4:6,:);
	end
    else
	error('Wrong number of arguments supplied');
    end
    
end

function [H, T1, T2] = homography(varargin)
    
    [x1, x2] = layout(varargin(:));

    [x1, T1] = normalise2dpts(x1);
    [x2, T2] = normalise2dpts(x2);
    
   
    Npts = length(x1);
    A = zeros(3*Npts,9);
    
    O = [0 0 0];
    for n = 1:Npts
	X = x1(:,n)';
	x = x2(1,n); y = x2(2,n); w = x2(3,n);
	A(3*n-2,:) = [  O  -w*X  y*X];
	A(3*n-1,:) = [ w*X   O  -x*X];
	A(3*n  ,:) = [-y*X  x*X   O ];
    end
    
    [U,D,V] = svd(A,0); 
    H = reshape(V(:,9),3,3)';
    
    H = T2\H*T1;
end

function [newpts, T] = normalise2dpts(pts)

    if size(pts,1) ~= 3
        error('pts must be 3xN');
    end
    
    finiteind = find(abs(pts(3,:)) > eps);
    
    if length(finiteind) ~= size(pts,2)
        warning('Some points are at infinity');
    end
    
    pts(1,finiteind) = pts(1,finiteind)./pts(3,finiteind);
    pts(2,finiteind) = pts(2,finiteind)./pts(3,finiteind);
    pts(3,finiteind) = 1;
    
    c = mean(pts(1:2,finiteind)')';            
    newp(1,finiteind) = pts(1,finiteind)-c(1);
    newp(2,finiteind) = pts(2,finiteind)-c(2);
    
    dist = sqrt(newp(1,finiteind).^2 + newp(2,finiteind).^2);
    meandist = mean(dist(:));
    scale = sqrt(2)/meandist;
    
    T = [scale   0   -scale*c(1)
         0     scale -scale*c(2)
         0       0      1      ];
    
    newpts = T*pts;
end

function H = computeDLT(uT, vT, u, v)
unit=ones(6,1);
p1=[u v unit]';
p2=[uT vT unit]';
[p1_norm, p1_trans]= normalise2dpts(p1);
[p2_norm, p2_trans]= normalise2dpts(p2);


A=[];
tempA=[];
for i=1:6
    tempA=[zeros(1,3) (-1)*p2_norm(:,i)' p1_norm(2,i)*p2_norm(:,i)';
        p2_norm(:,i)' zeros(1,3) -p1_norm(1,i)*p2_norm(:,i)'];
    A=[A ;tempA];
end

[U,S,V]=svd(A);
h=V(:,end);
H=reshape(h,[3, 3])';

H = inv(p1_trans)*H*(p2_trans);
end

function line=tls(data)
    x = data(1, :);
    y = data(2, :);

    k=(y(1)-y(2))/(x(1)-x(2));
    a=sqrt(1-1/(1+k^2));
    b=sqrt(1-a^2);

    if k>0
       b=-b; 
    end
    
    c=-a*x(1)-b*y(1);
    line=[a b c];
end
