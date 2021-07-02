
function trial()
    
close all;
clear all;
clc;
   
    N = 10;             % #[input values]
    H = 5;             % #[hidden nodes]
    t = 0:1/(N):1;    % [input values]
    
  
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

[theta,cost] = fminunc(@(p)(costFunction(t,p,H)),initial_param,options);
 
     toc;
%============output illustration============%
    
 
  disp("Error(Cost):");
 [cost,nx,np,nu] = costFunction(t,theta,H);
    
     
    subplot(2,2,1); plot(t,1 + nx .* (t),'r-o');
    hold on;
    plot(t,analyticx(t), 'b-x');
    xlabel('t');
    ylabel('x(t)');
    legend('Exact (x) and approximated (o) solution for function x(t)');
    title('State Variable');

    
   
    
   % figure;
    subplot(2,2,2);plot(t,np .* ((t)-1),'g-o');
    hold on;
    
    plot(t,analyticp(t), 'b-x');
    xlabel('t');
    ylabel('p(t)');
    title('Adjoint Variable');
     legend('Exact (x) and approximated (o) solution for function p(t)');
    
  
    
   % figure;
  subplot(2,2,3);   plot(t,nu,'m-o');
    hold on;
    plot(t,analyticu(t), 'b-x');
    xlabel('t');
    ylabel('u(t)');
    title('Control Variable');
     legend('Exact (x) and approximated (o) solution for function u(t)');
    
    disp(cost);

end