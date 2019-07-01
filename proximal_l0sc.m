function [x,s,objs,times] = proximal_l0sc(y,D,x0,lambda,maxIter,verbose,thr)

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

supp0 = support(x0);
cur_obj = compute_obj_robust(y,D,x0,lambda);
s = max(lambda/cur_obj,1.0/L);

objs = [];
times = [];
lastprintlength = 0;
if verbose,
    lastprintlength = textprogressbar(-1,lastprintlength,'begin l0sc points: ');
end

tStart = tic;
while ( iter <= maxIter )
    %df = 2*XtX*(alpha-eye(n));
    %c = 2;
    %c = 100;
    
    %add robustness to noise and outlier
    df = DtD*x0-Dty;
    %c = 2*S(1);
    
    
    x_proximal = x0 - s*df;
    
    x = x_proximal;
    
    x(x.^2 < 2*lambda*s) = 0;
    
    obj_x = compute_obj_robust(y,D,x,lambda);
    obj_x0 = compute_obj_robust(y,D,x0,lambda);
    supp_x = support(x);
    supp_x0 = support(x0);
    s1 = s;
    max_trial = 8;
    trial = 0;
    while ((obj_x >= obj_x0 - delta*norm(x-x0)*norm(x-x0)) || (~all(ismember(supp_x,supp_x0)))) && (trial < max_trial)
        s1 = s1*beta;
        x_proximal = x0 - s1*df;
        x = x_proximal;
        x(x.^2 < 2*lambda*s1) = 0;
        obj_x = compute_obj_robust(y,D,x,lambda);
        supp_x = support(x);
        trial = trial + 1;
    end
    
%     supp_x = support(x);
%     supp_x0 = support(x0);
%     if ~all(ismember(supp_x,supp_x0))
%         fprintf('support shrinkage violates in iter %d\n', iter);
%     end
    
    
    err = errorCoef(x,x0);
    
    if verbose,
        fprintf('proximal_manifold: errors = [%1.1e], iter: %4.0f \n',err,iter);
    end
    
    x0 = x;
       
    tElapsed = toc(tStart);
    times = [times;tElapsed];
    [obj,l2err,spar_err] = compute_obj_robust(y,D,x,lambda);
    objs = [objs;obj];
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