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