function [LOOE,ERR] = acv_mlr(wV,X,Ycode,Np,lambda2)
%--------------------------------------------------------------------------
% acv_mlr.m: An approximate leave-one-out estimator of predictive likelihood
% for multinomial logistic regression with elastic net regularization
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute and return an approximate leave-one-out estimator (LOOE) and its
%    standard error of predivtive likelihood for multinomial logistic regression 
%    penalized by elastic net regularization. 
%
% USAGE:
%    [LOOE,ERR] = acv_mlr(wV,X,Ycode,Np,lambda2)
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
%    lambda2     Coefficient of the l2 regularizaiton term   
%
% OUTPUT ARGUMENTS:
%    LOOE        Approximate value of the leave-one-out estimator 
%
%    ERR         Approximate standard error of the leave-one-out estimator 
%
% DETAILS:
%    The following multinomial logistic regression penalized 
%    by the l1 + l2 norms (elastic net) is considered:
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
%                 u_{a}=x.w_{a}.
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
%    27 Oct. 2017: Original version was written.
%    27 Jul. 2018: Updated to include elastic net.
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
W=zeros(Nparam,1);
for ip=1:Np
    W(N*(ip-1)+[1:N])=wV(:,ip);      % Extended representation of weight vectors 
    u_all(:,ip)=X*wV(:,ip);          % Overlaps for all data       
end
p_all=prob_multinomial(u_all);     % All-class probabilities for all data
for ip=1:Np
    for jp=1:Np
        % Inter-class Hessian (vector rep. w.r.t data dimension for computation speed)
        F_all{ip}{jp}=(ip==jp)*p_all(:,ip)-p_all(:,ip).*p_all(:,jp); 
    end
end

% Active set 
A=find(W(:));                     % Active set
A_ipt=mod(A-1,N)+1;               % Input vector index of active set
A_cla=idivide(int32(A-1),N)+1;    % Class index of active set
N_A=size(A,1);                    % Size of Active set
ORDER=[1:N_A]';
for ip=1:Np
    flag_cla=(A_cla==ip);
    As_ord{ip}=ORDER(flag_cla);   % Order of active components of each class
    As_ipt{ip}=A_ipt(flag_cla);   % Input vector index of active set of each class
end

% Construct Hessian
H=zeros(Nparam,Nparam);
H_diag=zeros(Nparam,1);
for i=1:N_A
    H_diag(A(i))=sum( X(:,A_ipt(i)).*F_all{A_cla(i)}{A_cla(i)}.*X(:,A_ipt(i)) );
    for j=i+1:N_A
        H(A(i),A(j))=sum( X(:,A_ipt(i)).*F_all{A_cla(i)}{A_cla(j)}.*X(:,A_ipt(j)) );
    end
end
G=H(A,A)+H(A,A)'+diag(H_diag(A));                     % Active part of Hessian 
G=G+lambda2*eye(N_A);                                 % Contribution from l2 norm

% Inverse Hessian with zero mode removal
thre=1.0e-6;                                              % Threshold detecting zero modes
if lambda2 > thre
    Ginv_zmr=inv(G);
else
    [V D]=eig(G);                                         % Eigenvalue decomposition
    A_rel=find(diag(D)>thre);                             % Relevant modes
    Ginv_zmr=V(:,A_rel)*inv(D(A_rel,A_rel))*V(:,A_rel)';  % Inverse without zero modes 
end

% LOO factor
C=zeros(Np,Np,M);
for mu=1:M
    for ip=1:Np
        for jp=1+ip:Np
            C(ip,jp,mu)=X(mu,As_ipt{ip})*Ginv_zmr(As_ord{ip},As_ord{jp})*X(mu,As_ipt{jp})';
        end
    end    
    C(:,:,mu)=C(:,:,mu)+C(:,:,mu)';
    for ip=1:Np
         C(ip,ip,mu)=X(mu,As_ipt{ip})*Ginv_zmr(As_ord{ip},As_ord{ip})*X(mu,As_ipt{ip})';
    end
end

% Gradient
b_all=zeros(Np,M);
for ip=1:Np
    b_all(ip,:)=p_all(:,ip)'-Ycode(:,ip)';
end

% LOOE 
I=eye(Np);
for ip=1:Np
    for jp=1:Np
        F(ip,jp,:)=F_all{ip}{jp}(:);                  % Inter-class Hessian
    end
end
u_all_loo=zeros(M,Np);                                % LOO overlap
for mu=1:M
    % Approximate LOO overlap
    u_all_loo(mu,:)=u_all(mu,:)+( C(:,:,mu)*( (I-F(:,:,mu)*C(:,:,mu))\b_all(:,mu) ) )';
end
p_all_loo=prob_multinomial(u_all_loo);                % LOO likelihood
LOOE=-mean(log(sum(Ycode.*p_all_loo,2)));             % LOOE
ERR=std(log(sum(Ycode.*p_all_loo,2)))/sqrt(M);        % LOOE's error bar

end