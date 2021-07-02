function [cost1,pt,ut] = costFunction2(t,p,H)
    % x = vector of the input
    % w = weight of hidden nodes
    % b = bias of hidden nodes
    % v = weight of output nodes
     %wx=p(1:H);
    %bx=p(1+H:2*H);
    %vx=(p(1+(2*H):3*H))';
    wp=p(1+(3*H):4*H);
    bp=p(1+(4*H):5*H);
    vp=p(1+(5*H):6*H)';
    wu=p(1+(6*H):7*H);
    bu=p(1+(7*H):8*H);
    vu=p(1+(8*H):9*H)';

    nu = vu*sigmoid(wu* t + bu);
    nu_d = sum(vu * (wu .* sigmoid_d(wu * t + bu)));
    
   % nx = vx * sigmoid(wx * t + bx);
   % nx_d = vx * (wx .* sigmoid_d(wx * t + bx));
  
    
    np = vp * sigmoid(wp * t + bp);
    np_d = vp * (wp .* sigmoid_d(wp * t + bp));
    
    
    %assume trial solutions for x,p and u as follows;
  % xt=4+(t.*nx) ;
  % dxt=nx+(nx_d.*t);
   
   pt=(t-2).*np;
   dpt=np+((t-2).*np_d);
    
 %ut=(t<1.096)*2; 
 %disp(ut);
% ut=nu;
 %dut=nu_d;
  ut=(pt>=3)*2+(pt<3)*0; 
       cost1 = sum((dpt+2+pt).^2);
      
%cost2 = sum((dxt-xt-ut).^2);
  %cost=cost1;%+cost2;
end