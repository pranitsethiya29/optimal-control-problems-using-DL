function [cost,xt,pt,ut] = costFunction(t,wx,bx,vx,wp,bp,vp,wu,bu,vu)
    % x = vector of the input
    % w = weight of hidden nodes
    % b = bias of hidden nodes
    % v = weight of output nodes

    nu = vu*sigmoid(wu* t + bu);
    nu_d = sum(vu * (wu .* sigmoid_d(wu * t + bu))); %diff of nu
    
    nx = vx * sigmoid(wx * t + bx);
    nx_d = vx * (wx .* sigmoid_d(wx * t + bx)); %diff of nx
  
    
    np = vp * sigmoid(wp * t + bp);
    np_d = vp * (wp .* sigmoid_d(wp * t + bp));%diff of np
    
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