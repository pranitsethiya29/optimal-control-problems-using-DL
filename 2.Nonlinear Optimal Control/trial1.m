
function trial1()
    
close all;
clear all;
clc;
   
    N = 10;             % #[input values]
    H = 5;             % #[hidden nodes]
    t = 0:1/(N-1):1    % [input values]
    
  
    wx = randn(H,1);
    bx = zeros(H,1);
    vx =randn(1,H);
    
    wp = randn(H,1);
    bp = zeros(H,1);
    vp = randn(1,H);
    
    wu = randn(H,1);
    bu = zeros(H,1);
    vu = randn(1,H);
    initial_theta=[wx;bx;vx';wp;bp;vp';wu;bu;vu'];
 tic;
    options = optimset('Display','off','HessUpdate','bfgs');
    [theta,cost] = fminunc(@(p)(costFunctions(t,p,H)),initial_theta,options);
 
     toc;
%output illustration %
    
 
   %[cost,nx,np,nu] = costFunctions(t,wx,bx,vx,wp,bp,vp,wu,bu,vu);
  disp("Error:");
 [cost,nx,np,nu] = costFunctions(t,theta,H);
    
     
    subplot(2,2,1); plot(t,(0.5*t)+(t.*(1-t).*nx),'r-o');
    
   
    xlabel('t');
    ylabel('x(t)');
    legend('Exact (x) and approximated (o) solution for function x(t)');
    title('State Variable');

    
   
    
   % figure;
    subplot(2,2,2);plot(t,np,'g-o');
    xlabel('t');
    ylabel('p(t)');
    title('Adjoint Variable');
     legend('Exact (x) and approximated (o) solution for function p(t)');
    
  
    
   % figure;
  subplot(2,2,3);   plot(t,nu,'m-o');
 xlabel('t');
    ylabel('u(t)');
    title('Control Variable');
     legend('Exact (x) and approximated (o) solution for function u(t)');
    
    disp(cost);

end