function [U, ConsenX, obj] = CDNMF(X,param,options)
%% ===================== Parameters =====================
numClust = options.numClust ;
NITER = param.NITER;
v = param.v;
n = param.n;
c = param.c;
alpha = param.alpha;
beta = param.beta;
omega = param.omega;
delta = param.delta;

%% ===================== initialize =====================
U = cell(1,v);
V = cell(1,v);
W = cell(1,v);
Xcon = cell(1,v);
E = cell(1,v);
Ei = cell(1,v);
p1 = cell(1,v);
P1 = cell(1,v);
qj = cell(1,v);
q = cell(1,v);
Q = cell(1,v);
I = eye(c);
H = abs(rand(c,n));
Y = cell(1,v);
J = cell(1,v);
%%
for vIndex=1:v
    Pn = size(X{vIndex},1);
    P1{vIndex} = eye(Pn);
    Q{vIndex} = eye(Pn);
    ni = size(X{vIndex},2);
    U{vIndex} = abs(rand(Pn,c));
    V{vIndex} = abs(rand(c,ni));
    W{vIndex} = abs(rand(c,c));
end

for i=1:v 
    J{i} = rand(numClust,c);
    Y{i} = rand(numClust,n);
    YY = litekmeans(X{i}',numClust,'MaxIter', 100);
    Y{i} = ToM(YY,size(Y{i},1),size(Y{i},2));
end 

for i = 1:v
    Xcon{i} = X{i}';
end
ConsenX = DataConcatenate(Xcon);
%% ===================== updating =====================
for iter = 1:NITER

 %% update U
    for iterv = 1:v
        U{iterv} = U{iterv}.*((P1{iterv}*X{iterv}*V{iterv}'+omega*U{iterv})./(P1{iterv}*U{iterv}*V{iterv}*V{iterv}'+beta*Q{iterv}*U{iterv}+omega*U{iterv}*U{iterv}'*U{iterv}+eps));
    end
 %% update V
    for iterv = 1:v
        V{iterv} = V{iterv}.*((U{iterv}'*P1{iterv}*X{iterv}+2*alpha*W{iterv}*H)./(U{iterv}'*P1{iterv}*U{iterv}*V{iterv}+2*alpha*V{iterv}+eps));
    end
 %% construct l_21 norm matrix
     for iterv = 1:v
        E{iterv} = X{iterv}-U{iterv}*V{iterv};
        Ei{iterv} = sqrt(sum(E{iterv}.*E{iterv},2)+eps);
        p1{iterv} = 0.5./Ei{iterv};
        P1{iterv} = diag(p1{iterv});
  
        qj{iterv} = sqrt(sum(U{iterv}.*U{iterv},2)+eps);
        q{iterv} = 0.5./qj{iterv};
        Q{iterv} = diag(q{iterv});
    end
 %% update W
    for iterv = 1:v
        [svd_U,~,svd_V] = svd(alpha*H*V{iterv}','econ');
        W{iterv} = svd_V*I*svd_U';
    end
 %% update H
    sum1 = zeros(c,n);
    sum2 = zeros(c,n);
    for sumv = 1:v
        sum1 = sum1+2*alpha*W{sumv}'*V{sumv}+2*delta*J{sumv}'*Y{sumv};
        sum2 = sum2+2*alpha*W{sumv}'*W{sumv}*H+2*delta*J{sumv}'*J{sumv}*H;
    end 
    % update H
    H = H.*((sum1)./(sum2+eps));
 %% undate J
    for iterv = 1:v
         J{iterv} = J{iterv}.*((2*delta*Y{iterv}*H')./(2*delta*J{iterv}*H*H'+eps));
    end
%% ===================== calculate obj =====================
    tempobj=0;
    for objIndex=1:v
        Term1 = trace((X{objIndex}-U{objIndex}*V{objIndex})'*P1{objIndex}*(X{objIndex}-U{objIndex}*V{objIndex}));
        Term2 = (norm(V{objIndex}-W{objIndex}*H,'fro')).^2;
        Term3 = trace((Y{objIndex}-J{objIndex}*H)'*(Y{objIndex}-J{objIndex}*H));
        Term4 = sum(sqrt(sum(U{objIndex}.*U{objIndex},2)));
        Term5 = trace((U{objIndex}'*U{objIndex}-I)*(U{objIndex}'*U{objIndex}-I)');
                tempobj=tempobj+Term1+alpha*Term2+delta*Term3+beta*Term4+omega*Term5;
    end
    obj(iter) = tempobj;  
    if iter == 1
        err = 0;
    else
        err = obj(iter)-obj(iter-1);
    end
    
    fprintf('iteration =  %d:  obj: %.4f; err: %.4f  \n', ...
        iter, obj(iter), err);
    if (abs(err))<1e+0
        if iter > 15
            break;
        end
    end
end










