function [LOOE,ERR] = saacv_mlr(wV,X,Ycode,Np,lambda2)
%--------------------------------------------------------------------------
% saacv_mlr.m: A further simplified approximation of 
% a leave-one-out estimator of predictive likelihood 
% for multinomial logistic regression with elastic net regularization
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute and return an very simplified approximation of 
%    a leave-one-out estimator (LOOE) and its standard error 
%    of predivtive likelihood for multinomial logistic regression 
%    penalized by elastic net regularization (l1 norm and l2 norm). 
%
% USAGE:
%    [LOOE,ERR] = saacv_mlr(wV,X,Ycode,Np,lambda2)
%
% INPUT ARGUMENTS:
%    wV          Weight vectors (N*Np dimensional vector). 
%                N is feature vector dimensionality
%                Np is the number of classes
%
%    X           Input feature matrixs (M*N dimensional matrix)
%
%    Ycode       M*Np dimensional binary matrix representing
%                the class to which the correponding feature vector belongs  
%
%    lambda2     Coefficient of the l2 regularizaiton term. Default value is zero.   
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
%                \hat{w}=argmin_{{w_a}_a^{Np}}
%                        { -\sum_{\mu}llkh({w_a}_a^{Np}|(y_{\mu},x_{\mu}))
%                                 + lambda*\sum_{a}^{Np}||w_a||_1 
%                                 + (1/2)*lambda_2*\sum_{a}^{Np}||w_a||_2^2},
%
%    where llkh=log\phi is the log likelihood of multinomial logistic map
%    \phi:
%
%                \phi(w|(y,x))=e^{u_{y}}/\sum_a e^{u_{a}}
%
%    where
%
%                 u_{a}=x.w_{a}
%
%    The leave-one-out estimator (LOOE) of a predictive likelihood is
%    defined as the 
%
%                LOOE=-\sum_{\mu}llkh({\hat{w}^{\backslash \mu}_a}_a^{Np}|(y_{\mu},x_{\mu}))/M,
%
%    where \hat{w}^{\backslash \mu}_a is the solution of the above 
%    minimization problem without the mu-th llkh term. 
%    This LOO solution \hat{w}^{\backslash \mu}_a is approximated 
%    from the full solution \hat{w}_a, yielding an approximate LOOE.
%
%
% REFERENCES:
%    Tomoyuki Obuchi and Yoshiyuki Kabashima 
%    ********************************************
%    arXiv:1711.05420
%
% DEVELOPMENT:
%    28 Oct. 2017: Original version was written.
%    27 Jul. 2018: Updated to include elastic net 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameter
[M,N]=size(X);
[M2 Np2]=size(Ycode);
[N2 Np3]=size(wV);
if nargin < 4
    error('four input arguments needed');
end
if Np ~= Np2
    error('the class number is inconsistent between the third and fourth arguments');
end
if Np ~= Np3
    error('the class number is inconsistent between the first and fourth arguments');
end
if M ~= M2
    error('data size is inconsistent between the second and third arguments');
end
if N ~= N2
    error('feature dimensionality is inconsistent between the first and second arguments');
end
if nargin < 4
    error('four input arguments needed.');
end
if nargin < 5 || isempty(lambda2)
    lambda2 = 0;
end
Nparam=N*Np;

% Preparation 
X2=X.^2;
mX2=mean(mean(X2));
for ip=1:Np
    u_all(:,ip)=X*wV(:,ip);          % Overlaps for all data       
end
p_all=prob_multinomial(u_all);       % All-class probabilities for all data
F=zeros(Np,Np,M);                    % Inter-class Hessian
for ip=1:Np
    for jp=1:Np
        F(ip,jp,:)=(ip==jp)*p_all(:,ip)-p_all(:,ip).*p_all(:,jp); 
    end
end

% Active set
for i=1:N
    A{i}=find(wV(i,:));
end

% SA approximation of LOO factor C
% Initialization
thre=10^(-6);
ERR=100;
I=eye(Np);
C_SA=zeros(Np,Np);                   % Projected susceptibility
chi_pre=zeros(Np,Np,N);              % Susceptibility 
chi=zeros(Np,Np,N);                  % Susceptibility  
for i=1:N
    chi(A{i},A{i},i)=1/mX2;
end

gamma0=0.1;
count=0;
theta=1.0e-6;
% Main loop computing C
while ERR > theta
    chi_pre=chi;
    gamma=min(0.9,gamma0+count*0.01);

    % Compute R
    R=lambda2*eye(Np);
    C_SA=mX2*sum(chi,3);
    for mu=1:M
        R=R+mX2*( I+F(:,:,mu)*C_SA )\F(:,:,mu);
    end

    % Update chi
    if lambda2>thre
        for i=1:N
            Rinv_zmr=inv(R(A{i},A{i}));
            chi(A{i},A{i},i)=gamma*chi_pre(A{i},A{i},i)+(1-gamma)*Rinv_zmr;
        end
    else        
        for i=1:N
            [V,D]=eig(R(A{i},A{i}));
            DV=diag(D);
            A_D=find(DV>thre);
            Rinv_zmr=V(:,A_D)*inv(D(A_D,A_D))*V(:,A_D)'; % Zero-mode-removed inverse of R
            chi(A{i},A{i},i)=gamma*chi_pre(A{i},A{i},i)+(1-gamma)*Rinv_zmr;
        end
    end

    % Error of chi
    ERR=0;
    for i=1:N
        ERR=ERR+norm(chi(A{i},A{i},i)-chi_pre(A{i},A{i},i),'fro');
    end
    ERR=ERR/N;
    
    count=count+1;
end
    
% Gradient
b_all=zeros(Np,M);
for ip=1:Np
    b_all(ip,:)=p_all(:,ip)'-Ycode(:,ip)';
end

% LOOE 
u_all_loo=zeros(M,Np);                                % LOO overlap
for mu=1:M
    % Approximate LOO overlap
    u_all_loo(mu,:)=u_all(mu,:)+(C_SA*b_all(:,mu))';
end
p_all_loo=prob_multinomial(u_all_loo);                % LOO likelihood
LOOE=-mean(log(sum(Ycode.*p_all_loo,2)));             % LOOE
ERR=std(log(sum(Ycode.*p_all_loo,2)))/sqrt(M);        % LOOE's error bar

end