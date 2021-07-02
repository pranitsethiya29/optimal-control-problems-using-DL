function [cost,nx,np,nu] = costFunction(t,p,H)
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
    nu_d = sum(vu * (wu .* sigmoid_d(wu * t + bu)));
    
    nx = vx * sigmoid(wx * t + bx);
    nx_d = vx * (wx .* sigmoid_d(wx * t + bx));
  
    
    np = vp * sigmoid(wp * t + bp);
    np_d = vp * (wp .* sigmoid_d(wp * t + bp));
    
     % cost1 = sum(((t.*nx_d)+nx-nu).^2);
   %cost2 = sum((2*(1+(t.*nx))+((t-1).*np_d)+np).^2);
    % cost3 = sum(((2*nu)+(np.*(t-1))).^2);
    
    %assume trial solutions for x,p and u as follows;
   xt=1+(t.*nx) ;
   dxt=nx+(nx_d.*t);
   
   pt=(t-1).*np;
   dpt=np+((t-1).*np_d);
    
   ut=nu; 
   dut=nu_d;
   
    cost1 = sum((dxt-ut).^2);             
   cost2 = sum(((2*xt)+dpt).^2);
    cost3 = sum(((2*ut)+pt).^2);
    
 cost=cost1+cost2+cost3;
end