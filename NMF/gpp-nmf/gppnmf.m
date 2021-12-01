function [D,H,XD,XH] = gppnmf(Y, L, varargin)
% GPPNMF Non-negative matrix factorization with Gaussian process prior
%
% Usage
%   [D,H,XD,XH] = gppnmf(Y, L, [options]);
%
% Input
%   Y               Data matrix
%   L               Number of factors
%   options
%     .maxIter      Number of iterations
%     .LinkFun      Function handle to link function.
%                     Use .DLinkFun and .HLinkFun to specify different 
%                     link functions for D and H.
%     .LinkPar      Cell array of parameters to pass to LinkFun in 
%                     addition to data.
%                     Use .DLinkPar and .HLinkPar to specify different 
%                     parameters for D and H.
%     .XD and .XH   Initial value of underlying Gaussian process.
%     .CD and .CH   Cholesky factorization of covariance matrices.
%     .alpha        Overall scale factor
%     .sigma        Noise variance
%
% Output
%   D, H            Non-negative factorization matrices               
%   XD, XH          Gaussian factorization matrices

% Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk


% Parse arguments
M = size(Y,1);
N = size(Y,2);
opts = mgetopt(varargin); 
maxIter = mgetopt(opts, 'maxIter',100);
XD = mgetopt(opts, 'XD', randn(L,M)/1e3);
XH = mgetopt(opts, 'XH', randn(L,N)/1e3);
Dgamma = mgetopt(opts, 'Dgamma', 1);
Hgamma = mgetopt(opts, 'Hgamma', 1);
CD = mgetopt(opts, 'CD', eye(M), 'dim', [M M]);
CH = mgetopt(opts, 'CH', eye(N), 'dim', [N N]);
LinkFun = mgetopt(opts, 'LinkFun', @linkFunExponential);
DLinkFun = mgetopt(opts, 'DLinkFun', LinkFun);
HLinkFun = mgetopt(opts, 'HLinkFun', LinkFun);
LinkPar = mgetopt(opts, 'LinkPar', {0, 1, 1});
DLinkPar = mgetopt(opts, 'DLinkPar', LinkPar);
HLinkPar = mgetopt(opts, 'HLinkPar', LinkPar);
alpha = mgetopt(opts, 'alpha', 1);
sigma = mgetopt(opts, 'sigma', 1);

D = DLinkFun(XD*CD', DLinkPar{:});
H = HLinkFun(XH*CH', HLinkPar{:});

sst = sum(sum((Y-mean(mean(Y(:)))).^2));

disp('GPP-NMF: Non-negative Matrix Factorization with Gaussian Process Priors');

mprogress;
for k = 1:maxIter
    [D, Cost, GradD] = costFun(XD, H, Y', CD, alpha, sigma, DLinkFun, DLinkPar);
    if k==1
        ConjD = -GradD;
    else
        beta = max(0, -GradD(:)'*(-GradD(:)+GradDp(:))/(GradDp(:)'*GradDp(:)));
        ConjD = -GradD + beta*ConjD;
    end
    GradDp = GradD;    
    [D, XD, Dgamma] = lineSrc(XD, D, Cost, GradD, ConjD, Dgamma, ...
        H, Y', CD, alpha, sigma, DLinkFun, DLinkPar);
    
    [H, Cost, GradH] = costFun(XH, D, Y, CH, alpha, sigma, HLinkFun, HLinkPar);
    if k==1
        ConjH = -GradH;
    else
        beta = max(0, -GradH(:)'*(-GradH(:)+GradHp(:))/(GradHp(:)'*GradHp(:)));
        ConjH = -GradH + beta*ConjH;
    end
    GradHp = GradH;
    [H, XH, Hgamma] = lineSrc(XH, H, Cost, GradH, ConjH, Hgamma, ...
        D, Y, CH, alpha, sigma, HLinkFun, HLinkPar);

    mprogress(k/maxIter);
end


%--------------------------------------------------------------------------
% Cost function
function [H, Cost, Grad] = costFun(X, D, Y, C, alpha, sigma, linkFun, linkFunPar)
[H, H1] = linkFun(X*C', linkFunPar{:});
Yhat = alpha*D'*H;
Cost = .5*sigma^-2*sum((Y(:)-Yhat(:)).^2)+.5*sum(X(:).^2);
if nargout>=3
    Grad = alpha*sigma^-2*(H1.*(D*(Yhat-Y)))*C+X;
end


%--------------------------------------------------------------------------
% Line search
function [z, x, a] = lineSrc(x0, z0, c0, g0, p, a, varargin)
[za, ca] = costFun(x0+a*p, varargin{:});
good = ca<=c0;
maxK = 5;
k = 0;
if good
    while good && k<maxK
        k = k+1;
        x = x0+a*p;
        z = za;
        a = a*2;
        [za, ca] = costFun(x0+a*p, varargin{:});
        good = ca<=c0 && k<maxK;
    end
else
    while ~good && k<maxK
        k = k+1;
        a = a*.5;
        x = x0+a*p;
        [z, ca] = costFun(x, varargin{:});
        good = ca<=c0;
    end
    if ~good
        x = x0;
        z = z0;
    end
end


%--------------------------------------------------------------------------
% Link function
function [z, z1] = linkFunExponential(y, mu, sigma, lambda)
ee = 1e-12;
if nargout>=1
    psi = ee+.5-.5*erf((y-mu)/(sqrt(2)*sigma));
    z = max(-lambda^-1*log(psi),0); 
    z(isnan(z)) = 0;
end
if nargout>=2
    epsilon = ee+exp(-(y-mu).^2/(2*sigma^2));
    z1 = 1/(sigma*lambda*sqrt(2*pi))*(epsilon./psi);
end

%--------------------------------------------------------------------------
% Argument parser
function out = mgetopt(varargin)
% MGETOPT Parser for optional arguments
% 
% Usage
%   Get a parameter structure from 'varargin'
%     opts = mgetopt(varargin);
%
%   Get and parse a parameter:
%     var = mgetopt(opts, varname, default);
%        opts:    parameter structure
%        varname: name of variable
%        default: default value if variable is not set
%
%     var = mgetopt(opts, varname, default, command, argument);
%        command, argument:
%          String in set:
%          'instrset', {'str1', 'str2', ... }
%
% Example
%    function y = myfun(x, varargin)
%    ...
%    opts = mgetopt(varargin);
%    parm1 = mgetopt(opts, 'parm1', 0)
%    ...

% Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk

if nargin==1
    if isempty(varargin{1})
        out = struct;
    elseif isstruct(varargin{1})
        out = varargin{1}{:};
    elseif isstruct(varargin{1}{1})
        out = varargin{1}{1};
    else
        out = cell2struct(varargin{1}(2:2:end),varargin{1}(1:2:end),2);
    end
elseif nargin>=3
    opts = varargin{1};
    varname = varargin{2};
    default = varargin{3};
    validation = varargin(4:end);
    if isfield(opts, varname)
        out = opts.(varname);
    else
        out = default;
    end
    
    for narg = 1:2:length(validation)
        cmd = validation{narg};
        arg = validation{narg+1};
        switch cmd
            case 'instrset',
                if ~any(strcmp(arg, out))
                    fprintf(['Wrong argument %s = ''%s'' - ', ...
                        'Using default : %s = ''%s''\n'], ...
                        varname, out, varname, default);
                    out = default;
                end
            case 'dim'
                if ~all(size(out)==arg)
                    fprintf(['Wrong argument dimension: %s - ', ...
                        'Using default.\n'], ...
                        varname);
                    out = default;
                end
            otherwise,
                error('Wrong option: %s.', cmd);
        end
    end
end

%--------------------------------------------------------------------------
% Progress counter
function mprogress(n)
% MPROGRESS Display elapsed and remaining time of a for-loop
%
% Example:
%
% for n=1:N
%    ... do stuff ...
%    mprogress(n/N);
% end

% Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk
persistent m t0 c p tp
if nargin==0
    n = 0;
end
if isempty(m), m = inf; end
if isempty(p) || n<m, p = 0.001; end
if n-m>p || n<m || n==1
    % Start new counter
    if n<m
        t0 = cputime;
        c = '0%';
    % Update counter
    else
        fprintf('%c',8*ones(length(c)+1,1)); 
        c = sprintf('%0.f%% (%s) %s', ...
            n*100, mtime(cputime-t0), mtime((cputime-t0)*(1-n)/n));
    end
    disp(c);
    pause(0); 
    drawnow;
    if ~isempty(tp) && n>m
        p = p/(cputime-tp+eps);
    end
    m = n;
    tp = cputime;
end

function tstr = mtime(t)
if t<60*60
    tstr = sprintf('%02.f:%02.f', floor(t/60), mod(t,60));
elseif t<60*60*24
    tstr = sprintf('%02.f:%02.f:%02.f', floor(t/60/60), mod(floor(t/60),60), mod(t,60));
else
    tstr = sprintf('%0.f - %02.f:%02.f:%02.f', floor(t/60/60/24), mod(floor(t/60/60),24), mod(floor(t/60),60), mod(t,60));
end