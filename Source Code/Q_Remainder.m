n=65374;

for i=1:n
    if mod(n,i) == 0
        disp(i);
        i=i+1;
    end
end