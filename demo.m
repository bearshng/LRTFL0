msi_sz=size(noisy_msi);
[~,R]=estimateR(reshape(noisy_msi, msi_sz(1)*msi_sz(2),msi_sz(3))','additive','off');
p_subspace=max(p_subspace,3);
L=min(msi_sz(1),msi_sz(2));
A=randn(msi_sz(1),L*p_subspace);
B=randn(msi_sz(2),L*p_subspace);
C=randn(msi_sz(3),p_subspace);
lambda=3;
alpha=10/(sqrt(msi_sz(1)*msi_sz(2))); % for mixture  for gaussian  alpha=1/(sqrt(msi_sz(1)*msi_sz(2)));
tau=300/(sqrt(msi_sz(1)*msi_sz(2)));
[~,~,~,clean_msi_lrtfl0]=LRTFL0(noisy_msi,A,B,C,p_subspace,lambda,alpha,tau);
