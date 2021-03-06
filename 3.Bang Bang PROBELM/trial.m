
function trial()
    
close all;
clc;
   
    N = 10;             
    H = 5;             % #[hidden nodes]
   t = 0:1/(N-1):2 ;  % [input values]
   
    wx = randn(H,1);
    bx = zeros(H,1);
    vx =randn(1,H);
    
    wp = randn(H,1);
    bp = zeros(H,1);
    vp = randn(1,H);
    
    wu = randn(H,1);
    bu = zeros(H,1);
    vu = randn(1,H);
    initial_param=[wx;bx;vx';wp;bp;vp';wu;bu;vu'];
 tic;
 
 %========BFGS OPTIMIZATION ALGORITHM==========%
 
    options = optimset('Display','off','HessUpdate','bfgs');

[theta,~] = fminunc(@(p)(costFunction2(t,p,H)),initial_param,options);
    
 
 [cost1,pt,ut] = costFunction2(t,theta,H);
 [theta1,~] = fminunc(@(p)(costFunctionx(t,p,H,ut)),initial_param,options);
 [cost2,xt] = costFunctionx(t,theta1,H,ut);
   toc;
 %=================================================================================  
%disp("Error(Cost):");
 cost=cost1+cost2;

   
   t1 = 0:1/(N-1):1 ;
   t2 = 1:1/(N-1):2 ; 
    subplot(2,2,1); plot(t,xt,'r-o');
    hold on;
    
       plot(t1,analyticx(t1), 'b-x',t2,analyticx2(t2), 'b-x');
     
     
   
    
    xlabel('t');
    ylabel('x(t)');
    legend('Exact (x) and approximated (o) solution for function x(t)');
    title('State Variable');

    
   
    
   % figure;
    subplot(2,2,2);plot(t,pt,'g-o');
    hold on;
    
    plot(t,analyticp(t), 'b-x');
    xlabel('t');
    ylabel('p(t)');
    title('Adjoint Variable');
     legend('Exact (x) and approximated (o) solution for function p(t)');
    
  
    
   % figure;
  subplot(2,2,3);   plot(t,ut,'m-o');
    hold on;
    plot(t,analyticu(t), 'b-x');
    xlabel('t');
    ylabel('u(t)');
    title('Control Variable');
     legend('Exact (x) and approximated (o) solution for function u(t)');
    
     disp("Error(Cost):");
    disp(cost);

end