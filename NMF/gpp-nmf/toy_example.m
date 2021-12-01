%%
disp('Setting parameters');

%% Parameters
M = 100;
N = 200;
L = 2;
mu = 0;
sigma = 1;
sigmaN = 5;
lambda = 1;
ee = 1e-12;
betaD = 0.01;
betaH = 0.01;
betaD2 = 0.1;
betaH2 = 0.001;

%%
disp('Computing covariance matrices');

%% Covariance matrices
M1 = repmat(shiftdim((1:M),1),[1,M]);
M2 = repmat(shiftdim((1:M),0),[M,1]);
CD = chol(exp(-betaD*(M1-M2).^2)+ee*eye(M))';
N1 = repmat(shiftdim((1:N),1),[1,N]);
N2 = repmat(shiftdim((1:N),0),[N,1]);
CH = chol(exp(-betaH*(N1-N2).^2)+ee*eye(N))';
CD2 = chol(exp(-betaD2*(M1-M2).^2)+ee*eye(M))';
CH2 = chol(exp(-betaH2*(N1-N2).^2)+ee*eye(N))';

%%
disp('Generating data');

%% Generate random data
% randn('state', 125);
% rng(125);
Dg = linkFunRectGauss(randn(L,M)*CD',mu,sigma,lambda);
Hg = linkFunExponential(randn(L,N)*CH',mu,sigma,lambda);
Y = Dg'*Hg + sigmaN*randn(M,N);

%%
disp('Computing GPP-NMF with correct prior');

%% GPP-NMF
% randn('state', 1);
% randn('state', 1);
[D1,H1,XD,XH] = gppnmf(Y, L, 'CD', CD, 'CH', CH, 'sigma', sigmaN, ...
    'maxIter', 1000, ...
    'DLinkFun', @linkFunRectGauss, ...
    'HLinkFun', @linkFunExponential);

%%
disp('Computing GPP-NMF with incorrect prior');

%% GPP-NMF Incorrect
% randn('state', 1);
[D4,H4,XD4,XH4] = gppnmf(Y, L, 'CD', CD2, 'CH', CH2, 'sigma', sigmaN, ...
    'maxIter', 1000, ...
    'DLinkFun', @linkFunExponential, ...
    'HLinkFun', @linkFunRectGauss);

%%
disp('Computing CNMF');

%% cnmf solution
% rand('state', 1);
D2 = rand(L,M);
H2 = rand(L,N);
ee = 1e-6;
for k=1:1000
    D2 = max(D2.*(H2*Y')./(H2*H2'*D2),ee);
    D2 = D2./repmat(sqrt(sum(D2.^2,2)),[1,size(D2,2)])*12;
    H2 = max(H2.*(D2*Y)./(D2*D2'*H2),ee);
end

%%
disp('Computing LS-NMF');

%% nmf solution
% rand('state', 1);
D3 = rand(L,M);
H3 = rand(L,N);
Y3 = max(Y,0);
for k=1:1000
    D3 = D3.*(H3*Y3')./(H3*H3'*D3);
    D3 = D3./repmat(sqrt(sum(D3.^2,2)),[1,size(D3,2)])*12;
    H3 = H3.*(D3*Y3)./(D3*D3'*H3);
end

%%
disp('Displaying results');

%% Figure 1
figure('Colormap', [0 1 0;0.04762 1 0.04762;0.09524 1 0.09524;0.1429 1 0.1429;0.1905 1 0.1905;0.2381 1 0.2381;0.2857 1 0.2857;0.3333 1 0.3333;0.381 1 0.381;0.4286 1 0.4286;0.4762 1 0.4762;0.5238 1 0.5238;0.5714 1 0.5714;0.619 1 0.619;0.6667 1 0.6667;0.7143 1 0.7143;0.7619 1 0.7619;0.8095 1 0.8095;0.8571 1 0.8571;0.9048 1 0.9048;0.9524 1 0.9524;1 1 1;1 1 0.8;1 1 0.6;1 1 0.4;1 1 0.2;1 1 0;0.9928 0.881 0.04183;0.9856 0.7621 0.08366;0.9784 0.6431 0.1255;0.9754 0.5922 0.1434;0.9723 0.5412 0.1613;0.9692 0.4902 0.1793;0.9661 0.4392 0.1972;0.963 0.3882 0.2151;0.9599 0.3373 0.2331;0.9569 0.2863 0.251;0.9183 0.2627 0.2556;0.8797 0.2392 0.2601;0.8412 0.2157 0.2647;0.8026 0.1922 0.2693;0.7641 0.1686 0.2739;0.7255 0.1451 0.2784;0.6909 0.1382 0.2652;0.6564 0.1313 0.2519;0.6218 0.1244 0.2387;0.5873 0.1175 0.2254;0.5528 0.1106 0.2121;0.5182 0.1036 0.1989;0.4837 0.09673 0.1856;0.4491 0.08982 0.1724;0.4146 0.08291 0.1591;0.38 0.076 0.1458;0.3455 0.06909 0.1326;0.3109 0.06218 0.1193;0.2764 0.05528 0.1061;0.2418 0.04837 0.09281;0.2073 0.04146 0.07955;0.1727 0.03455 0.06629;0.1382 0.02764 0.05303;0.1036 0.02073 0.03978;0.06909 0.01382 0.02652;0.03455 0.006909 0.01326;0 0 0]);
cax = [-10 20];

subplot(3,2,1);
imagesc(1:N,1:M,Y);
axis xy; axis equal; axis tight;
caxis(cax);
title('Noisy data');

subplot(3,2,2);
imagesc(1:N,1:M,Dg'*Hg);
axis xy; axis equal; axis tight;
caxis(cax);
title('Underlying data');

subplot(3,2,3);
imagesc(1:N,1:M,D3'*H3);
axis xy; axis equal; axis tight;
caxis(cax);
title('LS-NMF');

subplot(3,2,4);
imagesc(1:N,1:M,D2'*H2);
axis xy; axis equal; axis tight;
caxis(cax);
title('CNMF');

subplot(3,2,5);
imagesc(1:N,1:M,D4'*H4);
axis xy; axis equal; axis tight;
caxis(cax);
title('GPP-NMF: Incorrect prior');

subplot(3,2,6);
imagesc(1:N,1:M,D1'*H1);
axis xy; axis equal; axis tight;
caxis(cax);
title('GPP-NMF: Correct prior');

%% Figure 2
figure;

subplot(5,2,1);
plot(1:M, flipud(Dg));
xlim([1 M]); ylim([0 5]);
title('Colums of D');
ylabel('Underlying data');

subplot(5,2,2);
plot(1:N, flipud(Hg));
xlim([1 N]); ylim([0 5]);
title('Rows of H');

subplot(5,2,3);
plot(1:M, D3);
xlim([1 M]); ylim([0 5]);
ylabel('LS-NMF');

subplot(5,2,4);
plot(1:N, H3);
xlim([1 N]); ylim([0 5]);

subplot(5,2,5);
plot(1:M, D2);
xlim([1 M]); ylim([0 5]);
ylabel('CNMF');

subplot(5,2,6);
plot(1:N, H2);
xlim([1 N]); ylim([0 5]);

subplot(5,2,7);
plot(1:M, D4);
xlim([1 M]); ylim([0 5]);
ylabel(sprintf('GPP-NMF:\nIncorrect prior'));

subplot(5,2,8);
plot(1:N, H4);
xlim([1 N]); ylim([0 5]);

subplot(5,2,9);
plot(1:M, D1);
xlim([1 M]); ylim([0 5]);
ylabel(sprintf('GPP-NMF:\nCorrect prior'));

subplot(5,2,10);
plot(1:N, H1);
xlim([1 N]); ylim([0 5]);

