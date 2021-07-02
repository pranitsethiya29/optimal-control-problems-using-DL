function output = analyticu(t)
  

for i=1:length(t)
    if(t(i)<=1.096)
              output(i)=2;
          else
              output(i)=0;
    end
end

              
end



%e^(-x/5)*sin(x);