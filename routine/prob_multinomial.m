function [prob] = prob_multinomial_u(uV)
[M,Np]=size(uV);
Z=sum(exp(uV),2);
for p=1:Np
    prob(:,p)=exp(uV(:,p))./Z;
end
 