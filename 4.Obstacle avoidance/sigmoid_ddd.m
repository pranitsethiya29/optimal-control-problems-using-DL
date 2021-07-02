function output = sigmoid_ddd(x)
       output = (exp(x).*(1 - 4*exp(x)+exp(2*x)))./((1 + exp(x)).^4);
%   