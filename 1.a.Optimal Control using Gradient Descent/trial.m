function trial()
    
close all;
clear all;
clc;
    % network parameters for x%
   
    N = 5 ;             % #[input values]
    H = 5;             % #[hidden nodes]
    t = 0:1/(N-1):1    % [input values]
    
    learning_rate = 0.02;
    num_iteration = 200 ;
    err_precision = 1e-3;
    

    %=====================x===================
    wx = randn(H,1);
    bx = zeros(H,1);
    vx =randn(1,H);

    grad_wx = zeros(H,1);
    grad_bx = zeros(H,1);
    grad_vx = zeros(1,H);
    
    %=====================p===================
    wp = randn(H,1);
    bp = zeros(H,1);
    vp = randn(1,H);
    
    grad_wp = zeros(H,1);
    grad_bp = zeros(H,1);
    grad_vp= zeros(1,H);
    
  
    
   %=====================u===================
    wu = randn(H,1);
    bu = zeros(H,1);
    vu = randn(1,H);
    
    grad_wu = zeros(H,1);
    grad_bu = zeros(H,1);
    grad_vu = zeros(1,H);

 

%================== trainning process-gradient descent ==========================%
    
    tic;
    syms wu_ bu_ vu_;
    syms wp_ bp_ vp_;
    syms wx_ bx_ vx_;
    for loop=1:num_iteration
        
        tmp_wx = [wx;wx_]; tmp_wx = tmp_wx(1:H);
        tmp_bx = [bx;bx_]; tmp_bx = tmp_bx(1:H);
        tmp_vx = [vx,vx_]; tmp_vx = tmp_vx(1:H);
        
        tmp_wp = [wp;wp_]; tmp_wp = tmp_wp(1:H);
        tmp_bp= [bp;bp_]; tmp_bp = tmp_bp(1:H);
        tmp_vp= [vp,vp_]; tmp_vp = tmp_vp(1:H);
        
        tmp_wu = [wu;wu_]; tmp_wu = tmp_wu(1:H);
        tmp_bu = [bu;bu_]; tmp_bu = tmp_bu(1:H);
        tmp_vu = [vu,vu_]; tmp_vu = tmp_vu(1:H);
        
        for i=1:H
            %gradient descent for parameters of x;
            tmp_wx(i) = wx_; tmp_bx(i) = bx_; tmp_vx(i) = vx_; 
            
            grad_wx(i) = subs(diff(costFunction(t,tmp_wx,bx,vx,wp,bp,vp,wu,bu,vu),wx_),wx_,wx(i));
            grad_bx(i) = subs(diff(costFunction(t,wx,tmp_bx,vx,wp,bp,vp,wu,bu,vu),bx_),bx_,bx(i));
            grad_vx(i) = subs(diff(costFunction(t,wx,bx,tmp_vx,wp,bp,vp,wu,bu,vu),vx_),vx_,vx(i));
            
            tmp_wx(i) = wx(i); tmp_bx(i) = bx(i); tmp_vx(i) = vx(i);
            
              %gradient descent for parameters of p;
            tmp_wp(i) = wp_; tmp_bp(i) = bp_; tmp_vp(i) = vp_;
            
            grad_wp(i) = subs(diff(costFunction(t,wx,bx,vx,tmp_wp,bp,vp,wu,bu,vu),wp_),wp_,wp(i));
            grad_bp(i) = subs(diff(costFunction(t,wx,bx,vx,wp,tmp_bp,vp,wu,bu,vu),bp_),bp_,bp(i));
            grad_vp(i) = subs(diff(costFunction(t,wx,bx,vx,wp,bp,tmp_vp,wu,bu,vu),vp_),vp_,vp(i));
            
            tmp_wp(i) = wp(i); tmp_bp(i) = bp(i); tmp_vp(i) = vp(i);
            
             %gradient descent for parameters of u;
            tmp_wu(i) = wu_; tmp_bu(i) = bu_; tmp_vu(i) = vu_;
            
            grad_wu(i) = subs(diff(costFunction(t,wx,bx,vx,wp,bp,vp,tmp_wu,bu,vu),wu_),wu_,wu(i));
            grad_bu(i) = subs(diff(costFunction(t,wx,bx,vx,wp,bp,vp,wu,tmp_bu,vu),bu_),bu_,bu(i));
            grad_vu(i) = subs(diff(costFunction(t,wx,bx,vx,wp,bp,vp,wu,bu,tmp_vu),vu_),vu_,vu(i));
            
            tmp_wu(i) = wu(i); tmp_bu(i) = bu(i); tmp_vu(i) = vu(i);
            
          
            
        end
         %updating parameters after every iteration
        wx = wx - learning_rate * grad_wx; 
        bx = bx - learning_rate * grad_bx; 
        vx = vx - learning_rate * grad_vx; 
        
        wp = wp - learning_rate * grad_wp; 
        bp = bp - learning_rate * grad_bp; 
        vp = vp - learning_rate * grad_vp; 
        
        wu = wu - learning_rate * grad_wu; 
        bu = bu - learning_rate * grad_bu; 
        vu = vu - learning_rate * grad_vu; 
        
        
        [cost,~,~,~] = costFunction(t,wx,bx,vx,wp,bp,vp,wu,bu,vu);
        disp(['loop: ',num2str(loop),' | ','error: ', num2str(cost)]);
        toc;
        
        if cost < err_precision
            break;
        end
    end
    % output illustration %
    
   [cost,xt,pt,ut] = costFunction(t,wx,bx,vx,wp,bp,vp,wu,bu,vu);
  
     
    subplot(2,2,1);plot(t,xt,'r-o');
    hold on;
    plot(t,analyticx(t), 'b-x');
    xlabel('t');
    ylabel('x(t)');
    legend('Exact (x) and approximated (o) solution for function x(t)');
    title('State Variable');

    
   
    
    
  subplot(2,2,2); plot(t,pt,'g-o');
    hold on;
    
    plot(t,analyticp(t), 'b-x');
    xlabel('t');
    ylabel('p(t)');
    title('Adjoint Variable');
     legend('Exact (x) and approximated (o) solution for function p(t)');
    
  
    
   
   subplot(2,2,3); plot(t,ut,'m-o');
    hold on;
    plot(t,analyticu(t), 'b-x');
    xlabel('t');
    ylabel('u(t)');
    title('Control Variable');
     legend('Exact (x) and approximated (o) solution for function u(t)');
    
    disp(cost);

end

