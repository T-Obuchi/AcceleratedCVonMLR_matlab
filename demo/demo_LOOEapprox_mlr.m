%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of approximate cross-validation in 
% multinomial logistic regression with the elastic net regularization. 
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
alpha=2;                  % Feature-to-data ratio
N=200;                    % Feature vector dimensionality
Np=8;                     % Number of classes
rho0=0.5;                 % Feature-vector density
sigmaN2=0.1;              % Noise strength
M=ceil(alpha*N);          % Data dimensionality
K=ceil(rho0*N);           % Nonzero-components number
sigmaW2=1/rho0;           % Approximately set feature-vector norm to sqrt(N)

% True fertures
w0=zeros(N,Np);
for ip=1:Np
    IND=randperm(N);
    S_A=sort(IND([1:K]));
    w0(S_A,ip)=sqrt(sigmaW2)*randn(K,1);                  % True features of each class
end

% Observed fertures and classes
X=zeros(M,N);                                             % Observed feature vector
Y=randi(Np,[M,1]);                                        % Observed classes
Ycode=zeros(M,Np);                                        % Binary representation of observed classes
for mu=1:M
    class=Y(mu);                                          % True class of mu-th observation 
    Ycode(mu,class)=1;                                    % Binary representation of class
    X(mu,:)=w0(:,class)/sqrt(N)+sqrt(sigmaN2)*randn(N,1); % Observation=True feature+Gaussian noise
end
X_std=standardize_matrix(X);

%%
% lambda
r_exp=0.1;
lambda1=100*10.^(-r_exp*[0:4:40]);
alpha_glmnet=0.5;
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

% Multinomial fit
tic;
fit   = glmnet(X_std,Y,'multinomial',options);
toc
tic;
CVfit = cvglmnet(X_std,Y,'multinomial',options);    
toc

%%
Llam=size(fit.lambda,1);
lambda1_tmp=lambda1(1:Llam);
lambda2_tmp=lambda2(1:Llam);
wV=zeros(N,Np,Llam);
for ip=1:Np
    for ilam=1:Llam
        wV(:,ip,ilam)=fit.beta{ip}(:,ilam);
    end
end

%%
LOOEV=zeros(Llam,1);
LOOEV_err=zeros(Llam,1);
tic; 
for ilam=1:Llam
    % Approximate CV
    [LOOEV(ilam),LOOEV_err(ilam)] = acv_mlr(wV(:,:,ilam),X_std,Ycode,Np,lambda2_tmp(ilam));
end
toc

LOOEV_SA=zeros(Llam,1);
LOOEV_SA_err=zeros(Llam,1);
tic; 
for ilam=1:Llam
    % SA approximation
    [LOOEV_SA(ilam),LOOEV_SA_err(ilam)] = saacv_mlr(wV(:,:,ilam),X_std,Ycode,Np,lambda2_tmp(ilam));
end
toc

%% Plot
llkh=zeros(Llam,1);
llkh_err=zeros(Llam,1);
uV=zeros(M,Np);
for ilam=1:Llam
    for ip=1:Np
        uV(:,ip)=X_std*wV(:,ip,ilam);           % Effective field
    end
    pV_all{ilam}=prob_multinomial(uV);          % Probabilities for all classes and data
    
    llkh(ilam)=-mean(log(sum(Ycode.*pV_all{ilam},2)));
    llkh_err(ilam)=std(log(sum(Ycode.*pV_all{ilam},2)))/sqrt(M);
end

figure; 
hold on;
errorbar(lambda1_tmp,LOOEV,LOOEV_err,'r+');
errorbar(lambda1_tmp,LOOEV_SA,LOOEV_SA_err,'m>');
errorbar(lambda1_tmp,CVfit.cvm/2,CVfit.cvsd/2,'b*');
errorbar(lambda1_tmp,llkh,llkh_err,'k<');
title(['Simulated data, Np=',num2str(Np),', N=',num2str(N)]);
xlabel('\lambda_1');
ylabel('Errors');
ylim([min(llkh),2*max(CVfit.cvm/2)]);
legend('acv','saacv','10-fold','Training','Location','Best');
set(gca,'XScale','Log')
set(gca,'YScale','Log')




