function [ P ] = prob_logit(u)
M=length(u);
P=[exp(-u) ones(M,1)]./(1+exp(-u));
end

