function [LOOE,ERR] = saacv_logit(w,X,Ycode,lambda2)
%--------------------------------------------------------------------------
% saacv_logit.m: A further simplified approximation of 
% a leave-one-out estimator of predictive likelihood 
% for logistic regression with elastic net regularization
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute and return an very simplified approximation of
%    a leave-one-out estimator (LOOE) and its standard error 
%    of predivtive likelihood for logistic regression penalized by l1 norm. 
%
% USAGE:
%    [LOOE,ERR] = saacv_logit(w,X,Ycode,lambda2)
%
% INPUT ARGUMENTS:
%    w           Weight vector (N dimensional vector). 
%                N is feature vector dimensionality
%
%    X           Input feature matrixs (M*N dimensional matrix)
%
%    Ycode       M*2 dimensional binary matrix representing
%                the class to which the correponding feature vector belongs  
%
%    lambda2     Coefficient of the l2 regularizaiton term   
%
% OUTPUT ARGUMENTS:
%    LOOE        Approximate value of the leave-one-out estimator 
%
%    ERR         Approximate standard error of the leave-one-out estimator 
%
% DETAILS:
%    The following logistic regression penalized 
%    by the elastic net regularization (l1 + l2 norm) is considered:
%
%                \hat{w}=argmin_{w}
%                        { -\sum_{\mu}llkh(w|(y_{\mu},x_{\mu}))
%                          + lambda*||w||_1 + (1/2)lambda2*||w||_2^2 },
%
%    where llkh=log\phi is the log likelihood of logit model:
%
%                \phi(w|(y,x))=(delta_{y,1}+delta_{y,2}e^{u})/(1+e^{u})
%
%    where
%
%                 u=x.w
%
%    The leave-one-out estimator (LOOE) of a predictive likelihood is
%    defined as the 
%
%                LOOE=-\sum_{\mu}llkh(\hat{w}^{\backslash \mu}|(y_{\mu},x_{\mu}))/M,
%
%    where \hat{w}^{\backslash \mu} is the solution of the above 
%    minimization problem without the mu-th llkh term. 
%    This LOO solution \hat{w}^{\backslash \mu} is approximated 
%    from the full solution \hat{w}, yielding an approximate LOOE.
%
%
% REFERENCES:
%    Tomoyuki Obuchi and Yoshiyuki Kabashima 
%    ********************************************
%    arXiv:1711.05420
%
% DEVELOPMENT:
%    1 Nov. 2017: Original version was written.
%    27 Jul. 2018: Updated to include elastic net.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameter
[M,N]=size(X);
[M2 Np]=size(Ycode);
[N2]=size(w);
if nargin < 3
    error('three input arguments needed');
end
if Np ~=2
    error('the class number in third argument is inconsistent: two class case is treated');
end
if M ~= M2
    error('data size is inconsistent between the second and third arguments');
end
if N ~= N2
    error('feature dimensionality is inconsistent between the first and second arguments');
end
if nargin < 4 || isempty(lambda2)
    lambda2 = 0;
end

% Preparation 
X2=X.^2;
mX2=mean(mean(X2));
u_all=X*w;
p_all=prob_logit(u_all);      % All-class probabilities for all data
F_all=p_all(:,1).*p_all(:,2); % Hessian 

% Active set 
thre=1e-6;
A=find(abs(w)>thre);          % Active set
N_A=size(A,1);

% SA approximation of LOO factor C
% Initialization
gamma=.5;
ERR=100;
chi=1/mX2;
theta=1.0e-6;
% Main loop computing C
while ERR > theta
    chi_pre=chi;
    C_SA=N_A*mX2*chi;
    R=lambda2+sum(F_all./(1+F_all*C_SA));
    chi=gamma*chi_pre+(1-gamma)/R/mX2;
    ERR=norm(chi-chi_pre);
end
C_SA=N_A*mX2*chi;

% Gradient
b_all=Ycode(:,1)-exp(-u_all)./(1+exp(-u_all));

% LOOE 
u_all_loo=u_all+C_SA*b_all;                          % LOO overlap
p_all_loo=prob_logit(u_all_loo);                     % LOO likelihood
LOOE=-mean(log(sum(Ycode.*p_all_loo,2)));            % LOOE
ERR=std(log(sum(Ycode.*p_all_loo,2)))/sqrt(M);       % LOOE's error bar

end