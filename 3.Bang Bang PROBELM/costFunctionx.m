function [cost2,xt] = costFunctionx(t,p,H,ut)
    wx=p(1:H);
    bx=p(1+H:2*H);
    vx=(p(1+(2*H):3*H))';
    
    nx = vx * sigmoid(wx * t + bx);
    nx_d = vx * (wx .* sigmoid_d(wx * t + bx));
    
    xt=4+(t.*nx) ;
    dxt=nx+(nx_d.*t);
  
cost2 = sum((dxt-xt-ut).^2);
end