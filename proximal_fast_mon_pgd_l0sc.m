function [x,s,objs,times] = proximal_fast_mon_pgd_l0sc(y,D,x0,lambda,maxIter,verbose,thr)

basic_nargins = 3;
if (nargin < basic_nargins+1)
    % default rl1graph regularization parameter
    lambda = 0.1;
end
if (nargin < basic_nargins+2)
    % default error thresholds to stop ADMM
    maxIter = 100;    
end
if (nargin < basic_nargins+3)
    % default error thresholds to stop ADMM
    verbose = false;
end
if (nargin < basic_nargins+4)
    % default error thresholds to stop ADMM
    thr = [1*10^-6 1*10^-5 1*10^-5]; 
end

DtD = D'*D;
Dty = D'*y;

supp_x0 = support(x0);
D_S = D(:,supp_x0);
[S] = svd(D_S);
L = S(1)^2;
%s = 1.0/L;

thr1 = thr(1);
err = 10*thr1;
beta = 0.1;
delta = 1e-3;

iter = 1;
x = x0;
z = x0;
t0 = 0;

suppx0 = support(x0);
cur_obj = compute_obj_robust(y,D,x0,lambda);
s = max(lambda/cur_obj,1.0/L);

objs = [];
times = [];
lastprintlength = 0;
if verbose,
    lastprintlength = textprogressbar(-1,lastprintlength,'begin fast_pgd_mon_l0sc points: ');
end

tStart = tic;
while ( iter <= maxIter )
    
    t = (sqrt(1+4*t0^2)+1)/2;
    u = x + t0/t*(z - x) + (t0-1)/t*(x - x0);
    
    supp_z = support(z);
    supp_u = support(u);
    
    %fprintf('%d',supp_x);
    
    %project support of u to that of x
    u(setdiff(supp_u,supp_z)) = 0;
    w = u;
    
    df = DtD*w-Dty;
    %c = 2*S(1);
    
    %compute s
%     suppx = support(x);
%     if all(ismember(suppx,suppx0)) && (length(suppx) < length(suppx0))
%         D_S = D(:,supp_x0);
%         [S] = svd(D_S);
%         L = S(1)^2;
%     end
%     cur_obj = compute_obj_robust(y,D,x,lambda);
%     s = max(lambda/cur_obj,1.0/L);
    
    w_proximal = w - s*df;
    
    z1 = w_proximal;
    
    z1(z1.^2 < 2*lambda*s) = 0;
    
    obj_z1 = compute_obj_robust(y,D,z1,lambda);
    obj_w = compute_obj_robust(y,D,w,lambda);
    supp_w = support(w);
    supp_z1 = support(z1);
    s1 = s;
    max_trial = 8;
    trial = 0;
    while ((obj_z1 >= obj_w - delta*norm(z1-w)*norm(z1-w)) || (~all(ismember(supp_z1,supp_w)))) && (trial < max_trial)
        s1 = s1*beta;
        w_proximal = w - s1*df;
        z1 = w_proximal;
        z1(z1.^2 < 2*lambda*s1) = 0;
        obj_z1 = compute_obj_robust(y,D,z1,lambda);
        supp_z1 = support(z1);
        trial = trial + 1;
    end

%     supp_w = support(w);
%     supp_z1 = support(z1);
%     if ~all(ismember(supp_z1,supp_w))
%         fprintf('support shrinkage violates in iter %d\n', iter);
%     end
    
    [obj_z1,~,~] = compute_obj_robust(y,D,z1,lambda);
    [obj_x,~,~] = compute_obj_robust(y,D,x,lambda);
    
    %if iter==1,
    %    x1 = z1;
    %else
    if obj_z1 <= obj_x,
        x1 = z1;        
    else
        x1 = x;        
    end
    %end
    
    err = errorCoef(x1,x);
    
    if verbose,
        fprintf('proximal_manifold: errors = [%1.1e], iter: %4.0f \n',err,iter);
    end
    
    x0 = x;
    x = x1;
    z = z1;
    
    t0 = t;
    
    %cW = .5*(abs(alpha)+abs(alpha'));
    %perf  = [perf; usencut(cW,k,tlabel)];
    %perf  = [perf; sc(k,cW,tlabel,opt.km_iter)];
    [obj,l2err,spar_err] = compute_obj_robust(y,D,x,lambda);
    objs = [objs;obj];
    tElapsed = toc(tStart);
    times = [times;tElapsed];
    %fprintf('accuracy is %1.1e, obj is %.5f \n', perf(end,1), obj(end));
    if verbose,
        fprintf('obj is %.9f, l2err is %.5f, spar_err is %.5f \n', obj,l2err,spar_err);
    end
    
    iter = iter+1;

end

if verbose,
    textprogressbar(1,lastprintlength,' Done');
end

end

function supp = support(x)
    supp = find((abs(x(:)) > eps));
end


function [obj,l2err,spar_err] = compute_obj_robust(y,D,x,lambda)
    l2err = 0.5*norm(y-D*x,'fro')^2;
    spar_err = lambda*sum(abs(x(:)) > eps);
    obj = l2err + lambda*spar_err;    
end