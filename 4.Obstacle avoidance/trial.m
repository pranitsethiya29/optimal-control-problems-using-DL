
function trial()
    

clc;
 
    N = 20;            % #[input values]
    H =5;             % #[hidden nodes]
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
    
    wx1 = randn(H,1);
    bx1 = zeros(H,1);
    vx1 =randn(1,H);
    
    wp1 = randn(H,1);
    bp1 = zeros(H,1);
    vp1 = randn(1,H);
    
    wu1 = randn(H,1);
    bu1 = zeros(H,1);
    vu1 = randn(1,H);
    
     wn1 = randn(H,1);
    bn1 = zeros(H,1);
    vn1 = randn(1,H);
     wn2 = randn(H,1);
    bn2 = zeros(H,1);
    vn2 = randn(1,H);
    initial_param=[wx;bx;vx';wp;bp;vp';wu;bu;vu';wx1;bx1;vx1';wp1;bp1;vp1';wu1;bu1;vu1';wn1;bn1;vn1';wn2;bn2;vn2'];
 tic;
 
 %========BFGS OPTIMIZATION ALGORITHM==========%
 
    options = optimset('Display','off','HessUpdate','bfgs');

[theta1,~] = fminunc(@(p,V)(costFunction(t,p,H)),initial_param,options);
 
     toc;
%============output illustration============%
    
 
  disp("Error(Cost):");
[cost,xt1,xt2,pt1,pt2,V,theta] = costFunction(t,theta1,H);
x=xt1;
y=xt2;

 figure(1)
th = linspace(0,2*pi,500);
x1 = sqrt(0.1)*cos(th)+0.4;
y1 = sqrt(0.1)*sin(th)+0.5;
x2 = sqrt(0.1)*cos(th)+0.8;
y2 = sqrt(0.1)*sin(th)+1.5;
plot(x,y);
hold on
plot(x1,y1,'r',x2,y2,'r');
hold off
xlabel('x');
ylabel('y');
%sqrt(d2xt1(1).^2+d2xt2(1).^2);
title(sprintf('Obstacle avoidance state variables'));
axis image
    
    disp(cost);

end