s=[1 0.8;0.8 1];
x=mvnrnd(zeros(10000,2),s);
subplot(2,1,1)
plot(x(:,1),x(:,2),'.');hold on;
axis equal;
grid on;
[V,D]=eig(s);
newx=x*V(:,2);
vd=V*D;
quiver(0,0,vd(1,1)*15,vd(2,1)*15);
quiver(0,0,vd(1,2)*3,vd(2,2)*3);
subplot(2,1,2)
hist(newx,50);