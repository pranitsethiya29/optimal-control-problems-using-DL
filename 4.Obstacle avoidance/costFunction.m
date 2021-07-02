function [cost,xt1,xt2,pt1,pt2,V,theta,d2xt1,d2xt2] = costFunction(t,p,H)
    % x = vector of the input
    % w = weight of hidden nodes
    % b = bias of hidden nodes
    % v = weight of output nodes
    wx1=p(1:H);             wx2=p(1+(9*H):10*H);     
    bx1=p(1+H:2*H);          bx2=p(1+(10*H):11*H);  
    vx1=(p(1+(2*H):3*H))';   vx2=(p(1+(11*H):12*H))';
    wp1=p(1+(3*H):4*H);      wp2=p(1+(12*H):13*H);    
    bp1=p(1+(4*H):5*H);      bp2=p(1+(13*H):14*H);   
    vp1=p(1+(5*H):6*H)';     vp2=p(1+(14*H):15*H)';  
    wu1=p(1+(6*H):7*H);          
    bu1=p(1+(7*H):8*H);          
    vu1=p(1+(8*H):9*H)'; 
     wu2=p(1+(15*H):16*H);          
    bu2=p(1+(16*H):17*H);          
    vu2=p(1+(17*H):18*H)'; 
    
    wn1=p(1+(18*H):19*H);  
    bn1=p(1+(19*H):20*H) ;          
    vn1=p(1+(20*H):21*H)';        
    bn2=p(1+(21*H):22*H);        
    wn2=p(1+(22*H):23*H);
    vn2=p(1+(23*H):24*H)';
    
    nn1 = vn1*sigmoid(wn1* t + bn1);
     nn2 =vn2*sigmoid(wn2* t + bn2);
    
   nu1 = vu1*sigmoid(wu1* t + bu1);
   % nu1_d = sum(vu * (wu .* sigmoid_d(wu * t + bu)));
      nu2 = vu2*sigmoid(wu2* t + bu2);
    %nu2_d = sum(vu * (wu .* sigmoid_d(wu * t + bu)));
   
    nx1 = vx1 * sigmoid(wx1 * t + bx1);
    nx1_d = vx1 * (wx1 .* sigmoid_d(wx1 * t + bx1));
     nx1_dd = (vx1 * (wx1.*wx1.*wx1 .* sigmoid_dd(wx1 * t + bx1)));
     nx1_ddd = (vx1 * (wx1.*wx1.*wx1 .* sigmoid_ddd(wx1 * t + bx1)));
    nx2 = vx2 * sigmoid(wx2 * t + bx2);
    nx2_d = vx2 * (wx2 .* sigmoid_d(wx2 * t + bx2));
     nx2_dd = (vx2 * (wx2 .*wx2.* sigmoid_dd(wx2 * t + bx2)));
   nx2_ddd =(vx2 * (wx2.*wx2.*wx2 .* sigmoid_ddd(wx2 * t + bx2)));
  
    
    np1 = vp1 * sigmoid(wp1 * t + bp1);
    np1_d = vp1 * (wp1 .* sigmoid_d(wp1 * t + bp1));
     np2 = vp2 * sigmoid(wp2 * t + bp2);
    np2_d = vp2 * (wp2 .* sigmoid_d(wp2 * t + bp2));
  
    
    
    T=1;
  
    
    
    %assume trial solutions for x,p and u as follows;
   xt1=(1.2*t)+((t.*(t-T)).*nx1) ;
   dxt1=(((2*t)-T).*nx1)+(nx1_d.*(t.*(t-T)))+1.2;
   d2xt1=2*nx1+(((2*t)-T).*nx1_d)+(((2*t)-T).*nx1_d)+(nx1_dd.*(t.*(t-T)));
   d3xt1=(2*nx1_d)+(2*nx1_d)+((((2*t)-T).*nx1_dd)*2)+(((2*t)-T).*nx1_dd)+(nx1_ddd.*(t.*(t-T)));
   
   xt2=(1.6*t)+((t.*(t-T)).*nx2);
   dxt2=(((2*t)-T).*nx2)+(nx2_d.*(t.*(t-T)))+1.6;
   d2xt2=2*nx2+(((2*t)-T).*nx2_d)+(((2*t)-T).*nx2_d)+(nx2_dd.*(t.*(t-T)));
   d3xt2=(2*nx2_d)+(2*nx2_d)+((((2*t)-T).*nx2_dd)*2)+(((2*t)-T).*nx2_dd)+(nx2_ddd.*(t.*(t-T)));
   
   pt1=np1;
   dpt1=(np1_d);
   pt2=np2;
   dpt2=(np2_d);
    
   
 ut1=nu1;
 ut2=nu2;
   
 theta=ut2;
 
 
            
 
    
  
    c1 = (((xt1-0.4).^2 )+((xt2-0.5).^2)-0.1); 
    dc11=2*(xt1-0.4);
    dc12=2*(xt2-0.5);
    c2 = (((xt1-0.8).^2 )+((xt2-1.5).^2)-0.1); 
    dc21=2*(xt1-0.8);
    dc22=2*(xt2-1.5);
    
 %   n1=(nn1.*(c1==0))+(0.*(c1>0));
  %  n2=(nn2.*(c2==0))+(0.*(c2>0));
    
 % cost3 = sum((dpt1+(2*d2xt1.*d3xt1)+(pt1.*d2xt1)+(n1.*dc11)+(n2.*dc21)).^2);
 % cost4 = sum((dpt2+(2*d2xt2.*d3xt2)+(pt2.*d2xt2)+(n1.*dc12)+(n2.*dc22)).^2);
  cost3 = sum((dpt1+(2*d2xt1.*d3xt1)+(pt1.*d2xt1)).^2);
  cost4 = sum((dpt2+(2*d2xt2.*d3xt2)+(pt2.*d2xt2)).^2);
  
    V=min((sqrt((d2xt1.^2)+(d2xt2.^2))));
   % theta=atan(xt2/xt1);
    
  cost1 = sum((dxt1-(V.*cos(theta))).^2);   
  cost2 = sum((dxt2-(V.*sin(theta))).^2);
  
  
  
cost5 = max((sum(c1)));
 cost6= max((sum(c2)));
   
  
 cost=cost1+cost2+cost3+cost4-(cost5+cost6);
end