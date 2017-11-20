function [A]=standardize_matrix(A0);
[M,N]=size(A0);

A=zeros(M,N);
for i=1:N
    av=mean(A0(:,i));
    sig=std(A0(:,i));
    A(:,i)=(A0(:,i)-av)/sig;
end

end
