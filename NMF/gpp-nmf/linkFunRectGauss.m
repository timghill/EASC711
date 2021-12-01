function [z, z1] = linkFunRectGauss(y, mu, sigma, lambda)
if nargout>=1
    z = (lambda*sqrt(2))*erfinv(.5+.5*erf((y-mu)/(sqrt(2)*sigma)));
end
if nargout>=2
    z1 = (lambda/(2*sigma))*exp(z.^2/(2*lambda^2)-y.^2/(2*sigma^2));
end