% low pass filter + Gaussian Probablity Distribution Fitting
% @ author: wen
% @ date: 20190225

%% load csv manually
%bdd    csvread('./bdd_aspect_ratio.csv');
%vnx    csvread('./vnx_aspect_ratio.csv');
%cal    csvread('./cal_aspect_ratio.csv');
ratio=csvread('./bdd_aspect_ratio.csv');
%% get hist
lower=1;
upper=800;
total=upper-lower+1;
hist_r=hist(ratio,3200);
hist_r_cut=hist_r(lower:upper);
%% fft & low pass & ifft
lp = cat(2,ones(1,total*1/40),zeros(1,total*38/40),ones(1,total*1/40));

w=fft(hist_r_cut);
w_lp=w.*lp;
hist_lp=ifft(w_lp);
hist_lp_abs=abs(hist_lp);
plot(hist_lp_abs);
title('Aspect Ratio Histrogram after Low Pass Filtering');

%%
csvwrite('./hist_lp_abs.csv', hist_lp_abs);

% end of file