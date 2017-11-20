function [LOOE,ERR] = saacv_mlr(wV,X,Ycode,Np)
%--------------------------------------------------------------------------
% saacv_mlr.m: A further simplified approximation of 
% a leave-one-out estimator of predictive likelihood 
% for multinomial logistic regression with l1 regularization
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute and return an very simplified approximation of 
%    a leave-one-out estimator (LOOE) and its standard error 
%    of predivtive likelihood for multinomial logistic regression 
%    penalized by l1 norm. 
%
% USAGE:
%    [LOOE,ERR] = saacv_mlr(wV,X,Ycode,Np)
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
%                                         + lambda*\sum_{a}^{Np}||w_a||_1 },
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
gamma=0.5;
ERR=100;
I=eye(Np);
C_SA=zeros(Np,Np);                   % Projected susceptibility
chi_pre=zeros(Np,Np,N);              % Susceptibility 
chi=zeros(Np,Np,N);                  % Susceptibility  
for i=1:N
    chi(A{i},A{i},i)=1/mX2;
end
% Main loop computing C
while ERR > 1.0e-6
    chi_pre=chi;

    % Compute R
    R=zeros(Np,Np);
    C_SA=mX2*sum(chi,3);
    for mu=1:M
        R=R+( I+F(:,:,mu)*C_SA )\F(:,:,mu);
    end        

    % Update chi
    for i=1:N
        [V,D]=eig(R(A{i},A{i}));
        DV=diag(D);
        A_D=find(DV>10^(-8));
        Rinv_zmr=V(:,A_D)*inv(D(A_D,A_D))*V(:,A_D)'; % Zero-mode-removed inverse of R
        chi(A{i},A{i},i)=gamma*chi_pre(A{i},A{i},i)+(1-gamma)*Rinv_zmr/mX2;
    end

    % Error of chi
    ERR=0;
    for i=1:N
        ERR=ERR+norm(chi(A{i},A{i},i)-chi_pre(A{i},A{i},i),'fro');
    end
    ERR=ERR/N;
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