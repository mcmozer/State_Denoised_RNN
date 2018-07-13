m = [.1 .1; .4 .2; .8 .6; .2 .8];
v = [.10 .10 .14 .18]*.75;
n_a = size(m,1);

[x,y] = ndgrid(-.2:.025:1.2);
z=zeros(size(x));
for i = 1: n_a
   z = z + sqrt(v(i)) .* exp(-((x-m(i,1)).^2+(y-m(i,2)).^2)./2./v(i)/v(i));
end
v=surf(x,y,-2*z)
set(gca,'cameraposition',[ -13.9982   -0.5845    3.3395])
axis off
set(v,'linewidth',.25)
print2pdf attractorLandscape.pdf
