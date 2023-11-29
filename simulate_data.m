%% Simulating forward model & noisy data
addpath('model_sim',genpath('../sampler'))
clc; clear; close all; 
rng('default'); seed = rng;

% Synthetic signal
G0=2.9e+5;
thickness=6e-4;
radius=2.5e-3;
par0=[G0;thickness;radius];
P=3; % no. of parameters
freq=(.2:.01:.4) * 1e+3;
[cp,freq]=forward_model(par0,0,freq,1);

% Producing noisy but smooth sample paths using GP
distmat = squareform(pdist(freq'));
kernelmat = exp(-.5 * distmat.^2 / 30^2);
rootmat = chol(kernelmat);

N=5;
D=length(cp);
eps=normrnd(0,2e-2,[D,N]);
noisy_cp = exp(rootmat'*eps) .* cp;

figure(2)
plot(freq,cp,'--k','LineWidth',2)
line(freq,noisy_cp)
xlabel('Freq (Hz)')
ylabel('Phase velocity (m/s)')
legend('Noiseless','FontSize',16)
ax = gca;
ax.FontSize = 16;

%% Model and algorithm parameters
hp = [1 1; 1 3; 1 3];
lb = [0.5e+5;5e-4;2e-3];
ub = [7.5e+5;6.5e-4;4e-3];
S = 400;
K = 5;
Wgrid = 2.^(-10:1:0);
prior_opt = struct('hp',hp,'lb',lb,'ub',ub);
forward_opt = struct('prec',.5 / geomean(var(noisy_cp,0,2)), 'freq',freq);

%% Running the algorithm
delete(gcp('nocreate'));
parpool(4);

tic
[samples,loss_mat,W,rsfb,rsfb_se,log_wv] = smc_sfb(noisy_cp,P,S,K,S/2,Wgrid,prior_opt,forward_opt);
Pcv = mean(D/2*log(pi/forward_opt.prec)-logsumexp((W-1) * loss_mat, 2));
Pcv_se = std(D/2*log(pi/forward_opt.prec)-logsumexp((W-1) * loss_mat, 2));
% Pcv = mean(D*log(2/forward_opt.prec)-logsumexp((W-1) * loss_mat, 2));
% Pcv_se = std(D*log(2/forward_opt.prec)-logsumexp((W-1) * loss_mat, 2));
[samples,loss_mat] = smc_full(samples,W,loss_mat,noisy_cp,K,prior_opt,forward_opt);
toc

save('simulated_experiment1.mat','freq','cp','noisy_cp','par0',...
    'samples','loss_mat','W','rsfb','rsfb_se','log_wv',...
    'Pcv','Pcv_se')

figure(3)
scatter3(par0(1),par0(2),par0(3),70,[0 0.4470 0.7410],'*','LineWidth',2)
hold on
scatter3(samples1(1,:),samples1(2,:),samples1(3,:),25,[0.8500 0.3250 0.0980],'filled')
scatter3(samples2(1,:),samples2(2,:),samples2(3,:),25,[0.9290 0.6940 0.1250],'filled')
legend('Truth','Squared error','L1','FontSize',16)
hold off
xlabel('Shear modulus (Pa)')
ylabel('Thickness (m)')
zlabel('Radius (m)')
ax = gca;
ax.FontSize = 16;

figure(4)
scatter(samples1(1,:),samples1(3,:),25,[0.8500 0.3250 0.0980],'filled')
hold on
scatter(samples2(1,:),samples2(3,:),25,[0.9290 0.6940 0.1250],'filled')
hold off
xline(par0(1),'LineStyle','--','Color',[0 0.4470 0.7410],'LineWidth',2)
yline(par0(3),'LineStyle','--','Color',[0 0.4470 0.7410],'LineWidth',2)
legend('Squared error','L1','FontSize',16)
text(2.5e+5,2.9e-3,{'Corr1=0.6','Corr2=0.6'},'FontSize',16)
xlabel('Shear modulus (Pa)')
ylabel('Radius (m)')
ax = gca;
ax.FontSize = 16;

dens_grid = .01:.01:1;
prior_grid = betapdf(dens_grid,1,3);
dens_grid1 = lb(1) + dens_grid .* (ub(1)-lb(1));
dens_grid2 = lb(2) + dens_grid .* (ub(2)-lb(2));
dens_grid3 = lb(3) + dens_grid .* (ub(3)-lb(3));
prior_grid1 = 1/(ub(1)-lb(1)) * ones(1,length(dens_grid));
prior_grid2 = prior_grid ./ (ub(2)-lb(2));
prior_grid3 = prior_grid ./ (ub(3)-lb(3));

figure(5)
tiledlayout(1,3)
nexttile
plot(dens_grid1,prior_grid1,'-b','LineWidth',2);
xline(par0(1),'LineStyle','--','LineWidth',2,'Color','k')
hold on
histogram(samples1(1,:),'Normalization','pdf','FaceColor',[0.8500 0.3250 0.0980])
hold off
xlim([lb(1) ub(1)])
xlabel('Shear modulus (Pa)')
ylabel('Density est.')
ax = gca;
ax.FontSize = 16;
legend('Prior','Truth','Samples','FontSize',16)

nexttile
plot(dens_grid2,prior_grid2,'-b','LineWidth',2);
xline(par0(2),'LineStyle','--','LineWidth',2,'Color','k')
hold on
histogram(samples1(2,:),'Normalization','pdf','FaceColor',[0.8500 0.3250 0.0980])
hold off
xlabel('Thickness (m)')
ylabel('Density est.')
ax = gca;
ax.FontSize = 16;
legend('Prior','Truth','Samples','FontSize',16)

nexttile
plot(dens_grid3,prior_grid3,'-b','LineWidth',2);
xline(par0(3),'LineStyle','--','LineWidth',2,'Color','k')
hold on
histogram(samples1(3,:),'Normalization','pdf','FaceColor',[0.8500 0.3250 0.0980])
hold off
xlabel('Radius (m)')
ylabel('Density est.')
ax = gca;
ax.FontSize = 16;
legend('Prior','Truth','Samples','FontSize',16)