N1=1000; N2=500; N3=2000;
X1=mvnrnd([0,0], [1,0;0,1], N1);
X2=mvnrnd([3,6], [1,-0.8;-0.8,1], N2);
X3=mvnrnd([10,6], [1,0.8;0.8,1], N3);
A1=X1(:,1);A2=X1(:,2);
B1=X2(:,1);B2=X2(:,2);
C1=X3(:,1);C2=X3(:,2);
S1=(N1-1)*cov(X1); S2=(N2-1)*cov(X2); S3=(N3-1)*cov(X3);
Sw=S1+S2+S3;
M1=mean(X1); M2=mean(X2); M3=mean(X3);
Mu=(N1*M1+N2*M2+N3*M3)/(N1+N2+N3);
Sb=N1*(M1-Mu)'*(M1-Mu)+N2*(M2-Mu)'*(M2-Mu) +N3*(M3-Mu)'*(M3-Mu);
J=inv(Sw)*Sb;  [V,D]=eig(J);

subplot(2,2,1)
plot(A1,A2,'r+');hold on;
plot(B1,B2,'g+');hold on;
plot(C1,C2,'b+');hold on;
quiver(0,0,V(1,1)*15,V(2,1)*15);hold on;
quiver(0,0,V(1,2)*10,V(2,2)*10);hold on;

subplot(2,2,3)
A=X1*V(:,1)*V(:,1)';
B=X2*V(:,1)*V(:,1)';
C=X3*V(:,1)*V(:,1)';
h1=histogram(A(:,1),20);hold on;
h1.FaceColor = 'r';
h2=histogram(B(:,1),20);hold on;
h2.FaceColor = 'g';
h3=histogram(C(:,1),20);hold on;
h3.FaceColor = 'b';

subplot(2,2,4)
A=X1*V(:,2)*V(:,2)';
B=X2*V(:,2)*V(:,2)';
C=X3*V(:,2)*V(:,2)';
h1=histogram(A(:,1),20);hold on;
h1.FaceColor = 'r';
h2=histogram(B(:,1),20);hold on;
h2.FaceColor = 'g';
h3=histogram(C(:,1),20);hold on;
h3.FaceColor = 'b';