%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of approximate cross-validation in 
% Logistic regression with the elastic net regularization. 
% By Tomoyuki Obuchi
% Origial version was written on 2017 Oct. 26.
% Updated on 2018 Jul. 26.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Method: 
%   See arXiv:1711.05420
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;

% Simulated data: Parameters
addpath('../routine/');
rng(1);
alpha=2;
N=400;
rho0=0.5;                 % Feature-vector density
M=ceil(alpha*N);
K=ceil(rho0*N);           
sigmaW2=1/rho0;           % Approximately set feature-vector norm to sqrt(N)

%@Input data
w0=zeros(N,1);
w0(1:K)=sqrt(sigmaW2)*randn(K,1);   % True feacher vector
X=standardize_matrix(randn(M,N));   % Random feature
u0=X*w0;

% Output data
Y=1./(1+exp(-u0))>0.5;
Ycode=[Y==0 Y==1];

%%
% lambda
r_exp=0.06;
lambda1=100*10.^(-r_exp*[0:40]);
alpha_glmnet=0.9;
lambda_glmnet=lambda1/(M*alpha_glmnet);
lambda2=lambda_glmnet*(1-alpha_glmnet)*M;

% Options for glmnet
%path(path,'../glmnet_matlab');      % Please add your place of "glment" to path.
options=glmnetSet();
options.lambda=lambda_glmnet;       % Setting lambda
options.alpha=alpha_glmnet;         % Setting alpha
options.intr=0;                     % Zeroing intercept
options.thresh=1.0e-8;              % Threshold for convergence
options.maxit=10^7;                 % Max iteration

% Binomial logit fit
tic;
fit = glmnet(X, Y, 'binomial', options);
t1=toc

tic;
CVfit = cvglmnet(X,Y,'binomial', options);    
t2=toc

%%
Llam=size(fit.lambda,1);
lambda1_tmp=lambda1(1:Llam);
lambda2_tmp=lambda2(1:Llam);
wV=zeros(N,Llam);
for ilam=1:Llam
    wV(:,ilam)=fit.beta(:,ilam);
end

LOOEV=zeros(Llam,1);
LOOEV_err=zeros(Llam,1);
tic; 
for ilam=1:Llam
    % Approximate CV
    [LOOEV(ilam),LOOEV_err(ilam)] = acv_logit(wV(:,ilam),X,Ycode,lambda2(ilam));
end
t3=toc

LOOEV_SA=zeros(Llam,1);
LOOEV_SA_err=zeros(Llam,1);
tic; 
for ilam=1:Llam
    % SA approximation
    [LOOEV_SA(ilam),LOOEV_SA_err(ilam)] = saacv_logit(wV(:,ilam),X,Ycode,lambda2(ilam));
end
t4=toc

%% Plot
llkh=zeros(Llam,1);
llkh_err=zeros(Llam,1);
for ilam=1:Llam
    u_all=X*wV(:,ilam);               % Effective field
    p_all=prob_logit(u_all);          % Probabilities for all classes and data
    llkh(ilam)=-mean(log(sum(Ycode.*p_all,2)));
    llkh_err(ilam)=std(log(sum(Ycode.*p_all,2)))/sqrt(M);
end

figure; 
hold on;
errorbar(lambda1_tmp,LOOEV,LOOEV_err,'r+');
errorbar(lambda1_tmp,LOOEV_SA,LOOEV_SA_err,'m>');
errorbar(lambda1_tmp,CVfit.cvm/2,CVfit.cvsd/2,'b*');
errorbar(lambda1_tmp,llkh,llkh_err,'k<');
title(['Simulated data, binary, N=',num2str(N)]);
xlabel('\lambda_1');
ylabel('Errors');
xlim([0.9*min(lambda1_tmp) 1.1*max(lambda1_tmp)]);
ylim([min(CVfit.cvm/2)/100,2]);
legend('Approx.','SA Approx','10-fold','Training error','Location','Best');
set(gca,'XScale','Log')
set(gca,'YScale','Log')



