function [LOOE,ERR] = acv_logit(w,X,Ycode)
%--------------------------------------------------------------------------
% acv_logit.m: An approximate leave-one-out estimator of predictive likelihood
% for logistic regression with l1 regularization
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute and return an approximate leave-one-out estimator (LOOE) and 
%    its standard error of predivtive likelihood for logistic regression 
%    penalized by l1 norm. 
%
% USAGE:
%    [LOOE,ERR] = acv_logit(w,X,Ycode)
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
% OUTPUT ARGUMENTS:
%    LOOE        Approximate value of the leave-one-out estimator 
%
%    ERR         Approximate standard error of the leave-one-out estimator 
%
% DETAILS:
%    The following multinomial logistic regression penalized by the l1 norm 
%    is considered:
%
%                \hat{w}=argmin_{w}
%                        { -\sum_{\mu}llkh(w|(y_{\mu},x_{\mu}))
%                                         + lambda*||w||_1 },
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

% Preparation 
u_all=X*w;
p_all=prob_logit(u_all);      % All-class probabilities for all data
F_all=p_all(:,1).*p_all(:,2); % Hessian 

% Active set 
thre=1e-8;
A=find(abs(w)>thre);          % Active set
N_A=size(A,1);

% Hessian
G=X(:,A)'*diag(F_all)*X(:,A);

% LOO factor
C=zeros(M,1);
C_r=G\X(:,A)';
for mu=1:M
    C(mu)=X(mu,A)*C_r(:,mu);
end

% Gradient
b_all=Ycode(:,1)-exp(-u_all)./(1+exp(-u_all));

% LOOE 
u_all_loo=u_all+C./(1-F_all.*C).*b_all;              % LOO overlap
p_all_loo=prob_logit(u_all_loo);                     % LOO likelihood
LOOE=-mean(log(sum(Ycode.*p_all_loo,2)));            % LOOE
ERR=std(log(sum(Ycode.*p_all_loo,2)))/sqrt(M);       % LOOE's error bar

end