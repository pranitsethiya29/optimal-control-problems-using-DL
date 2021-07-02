function [cost,nx,np,nu] = costFunctions(t,p,H)
    % x = vector of the input
    % w = weight of hidden nodes
    % b = bias of hidden nodes
    % v = weight of output nodes
     wx=p(1:H);
    bx=p(1+H:2*H);
    vx=(p(1+(2*H):3*H))';
    wp=p(1+(3*H):4*H);
    bp=p(1+(4*H):5*H);
    vp=p(1+(5*H):6*H)';
    wu=p(1+(6*H):7*H);
    bu=p(1+(7*H):8*H);
    vu=p(1+(8*H):9*H)';

    nu = vu*sigmoid(wu* t + bu);
    %nu_d = sum(vu * (wu .* sigmoid_d(wu * t + bu)));
    
    nx = vx * sigmoid(wx * t + bx);
    nx_d = vx * (wx .* sigmoid_d(wx * t + bx));
  
    
    np = vp * sigmoid(wp * t + bp);
    np_d = vp * (wp .* sigmoid_d(wp * t + bp));
 
    
   xt=(0.5*t)+(t.*(1-t).*nx);
   dxt=(0.5+((1-(2*t)).*nx))+((t.*(1-t).*nx_d));
   
   pt=np;
   dpt=np_d;
   
   ut=nu;
%   dut=nu_d; 
    
   cost1 = sum((dxt-(0.5*(xt.^2).*sin(xt))+(0.5*pt)).^2);
   cost2 = sum((dpt+(pt.*xt.*sin(xt))+(0.5*pt.*(xt.^2).*cos(xt))).^2);
   cost3 = sum(((2*ut)+pt).^2);
   cost=cost1+cost2+cost3;
   
end