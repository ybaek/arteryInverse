clc; clear; close all;
addpath('model')
addpath('../sampler')
addpath('../files/ProcessedDispersionInVivo/')
load('Dispersion_proposedProcessing_200_to_800Hz001.mat')
delete(gcp('nocreate'));
parpool(4);

freq = unique(freq_expt_left);
ind = find(freq >= 250 & freq <= 500);
freq = freq(ind);

prior_opt = struct(...
    'hp',[1 1;1 1;1 1],...
    'lb',[200e+3;.5e-3;1.5e-3],...
    'ub',[600e+3;1.2e-3;3.5e-3]...
);
grid = 2.^(-8:1:0);
P = 3;
S = 50;
K = 3;
emin = S/2;
%par_prev = zeros(P,10);
%cp_prev = zeros(length(freq),10);
samples_total = zeros(P,S,10);
W_total = zeros(1,10);
rsfb_total = zeros(length(grid),10);
for A = 1:10
    y = squeeze(cp_expt_left(ind,A,:));
    %par_prev(:,A) = [G0(A);thickness(A);dia(A)/2];
    %cp_prev(:,A) = forward_model(par_prev(:,A),0,freq,1);
    forward_opt = struct(...
    'alpha',0,... % No viscosity
    'freq',freq,...
    'prec', .5/median(mad(y,0,2))... % Other choices possible
    );
    tic
    [samples,loss_mat,W,rsfb,rsfb_se] = smc_sfb(y,P,S,emin,grid,prior_opt,forward_opt);
    [samples,loss_mat] = smc_full(samples,W,loss_mat,y,K,prior_opt,forward_opt);
    toc
    samples_total(:,:,A) = samples;
    W_total(A) = W;
    rsfb_total(:,A) = rsfb';
end    
save('invivo_inversion1.mat','samples_total','W_total','rsfb_total','par_prev','cp_prev')

%%
load('..\files\invivo_inversion1.mat')
seed =rng;
rind = randperm(400,6);
cp_sim = zeros(length(freq),6,10);
for j = 1:6
    for k = 1:10
%        par_jk = [samples_total(:,rind(j),k);par_prev(2,k);par_prev(3,k)];
        par_jk = [samples_total(:,rind(j),k)];
        cp_sim(:,j,k) = forward_model(par_jk,0,freq,1);
    end
end

figure(1)
tiledlayout(2,1)
nexttile
G0_mean = squeeze(mean(samples_total(1,:,:),2)) / 1e+3;
G0_std = squeeze(std(samples_total(1,:,:),0,2)) / 1e+3;
plot(1:10,G0_mean,'b-o')
xlabel('Pressure point')
ylabel('G0 (kPa)')
ylim([prior_opt.lb(1) prior_opt.ub(1)] / 1e+3)
hold on
plot(1:10,G0_mean + G0_std,'b-.o')
plot(1:10,G0_mean - G0_std,'b-.o')
plot(1:10,par_prev(1,:) / 1e+3,'k--o','LineWidth',1)
hold off
legend('Post. Mean','Post. SD','','Point est.')
title('Multiple parameters')

% tiledlayout(2,2)
% for j = 1:4
%     nexttile
%     scatter(log10(samples_total(1,:,j)), log10(samples_total(2,:,j)))
%     xlim(log10([prior_opt.lb(1) prior_opt.ub(1)]))
%     xlabel('log10(G0)')
%     ylabel('log10(Thickness)')
%     xline(log10(par_prev(1,j)),'LineWidth',1,'LineStyle','--','Color','k');
%     xline(log10(mean(samples_total(1,:,j))),'LineWidth',1,'LineStyle','--','Color','r');
%     yline(log10(par_prev(2,j)),'LineWidth',1,'LineStyle','--','Color','k');
%     yline(log10(mean(samples_total(2,:,j))),'LineWidth',1,'LineStyle','--','Color','r');
%     legend('','Posterior Mean','Point est.','FontSize',8)
%     title(strcat("Pressure point ",string(j)))
% end

figure(3)
t = tiledlayout(2,2);
for j = 1:4
    nexttile
    y = squeeze(cp_expt_left(ind,j,:));
    plot(freq,y,'LineWidth',2)
    xlim([min(freq) max(freq)])
    line(freq,cp_prev(:,j),'LineWidth',1,'LineStyle','--','Color','k')
    for k = 1:6
        line(freq,cp_sim(:,k,j),'LineStyle','-.','Color','b')
    end
    legend('','','','','','Point est.','Samples','FontSize',8)
    title(strcat("Pressure point ",string(j)))
end
title(t,'Multiple parameters')