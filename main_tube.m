clc; clear; close all;
addpath("model")
addpath("../sampler")
addpath(genpath("../files"))
load('RheospectrisDataForYoungsoo.mat')
delete(gcp('nocreate'));
parpool(12);

tube_ids = [1 3 4 6 7 8 9 10 11 12];

true_thick = 1e-3;
true_rad = 3e-3;
for i = 1:10
    id = tube_ids(i);
    filename = strcat("ProcessedDispersionAllTubes/ProcessedDispersion_tube",string(id),".mat");
    load(filename);

    freq = unique(freq_store) * 1e+3;
    ind1 = find(freq >= 300 & freq <= 500);
    ind2 = find(freq >= 900 & freq <= 1000);
    freq1 = freq(ind1);
    freq2 = freq(ind2);
	G0 = fitted_param(1,i) * 1e+3; % Rheospectris validation

    par_rheo = [G0;true_thick;true_rad];
    alpha_rheo = fitted_param(2,i);
    cp_rheo1 = forward_model(par_rheo,fitted_param(2,i),freq1,2);
    cp_rheo2 = forward_model(par_rheo,fitted_param(2,i),freq2,1);
    
    grid = 2.^(-8:1:0);
    P = 1;
    S = 400;
    K = 15;
    emin = S/2;
    prior_opt = struct(...
        'hp',[1,1],...
        'lb',10e+3,...
        'ub',200e+3...
    );

    samples_total = zeros(6,S);
    W_total = zeros(1,6);
    rsfb_total = zeros(6,length(grid));
    for A = 1:6
        Aind = ((A-1)*10+1):(A*10);
        y1 = cp_store(Aind,ind1)';
        y2 = cp_store(Aind,ind2)';
        y = [y1;y2];
        forward_opt = struct(...
            'alpha',alpha_rheo,...
            'freq1',freq1,...
            'freq2',freq2,...
            'thick',true_thick,...
            'rad',true_rad,...
            'prec',.5/median(mad(y,0,2))... % Other choices possible
        );
        tic
        [samples,loss_mat,W,rsfb] = smc_sfb(y,P,S,emin,grid,prior_opt,forward_opt);
        [samples,loss_mat] = smc_full(samples,W,loss_mat,y,K,prior_opt,forward_opt);
        toc
        samples_total(A,:) = samples;
        W_total(A) = W;
        rsfb_total(A,:) = rsfb;
    end
    filename = char(strcat("tube_inversion",string(id),".mat"));
    save(filename,'samples_total','W_total','rsfb_total','par_rheo','alpha_rheo','cp_rheo1','cp_rheo2')
end

%%
load('tube_inversion12.mat')
seed = rng;
rind = randperm(400,5);
cp_sim1 = zeros(length(freq1),5,6);
cp_sim2 = zeros(length(freq2),5,6);
for j = 1:5
    for k = 1:6
        parj = [samples_total(k,rind(j));par_rheo(2);par_rheo(3)];
        cp_sim1(:,j,k) = forward_model(parj,alpha_rheo,freq1,2);
        cp_sim2(:,j,k) = forward_model(parj,alpha_rheo,freq2,1);
    end
end

figure(1)
tiledlayout(4,3)
for j = 1:6
    nexttile
    histogram(samples_total(j,:) / 1e+3)
    xline(par_rheo(1) / 1e+3,'LineWidth',1,'LineStyle','--');
    title(strcat("Angle ",string(j)))
    xlabel('G0 (kPa)')
end
nexttile(7,[2,3])
histogram(samples_total / 1e+3)
xline(par_rheo(1) / 1e+3, 'LineWidth',1,'LineStyle','--');
xlabel('G0 (kPa)')

figure(2)
tiledlayout(3,2)
for j = 1:6
    nexttile
    y1 = cp_store(((j-1)*10+1):(j*10),ind1)';
    y2 = cp_store(((j-1)*10+1):(j*10),ind2)';
    plot(freq1,y1,'LineWidth',2)
    hold on
    plot(freq2,y2,'LineWidth',2)
    xlim([min(freq1) max(freq2)])
    line(freq1,cp_rheo1,'LineWidth',1,'LineStyle','--','Color','k')
    line(freq2,cp_rheo2,'LineWidth',1,'LineStyle','--','Color','k')
    for k = 1:5
        line(freq1,cp_sim1(:,k,j),'LineStyle','-.','Color','r')
        line(freq2,cp_sim2(:,k,j),'LineStyle','-.','Color','r')
    end
    title(strcat("Angle ",string(j)))
end

bias = zeros(1,10);
pstd = zeros(1,10);
coverage = zeros(1,10);
for j = 1:length(tube_ids)
    load(strcat("tube_inversion", string(tube_ids(j)), ".mat"))
    bias(j) = abs(mean(samples_total,'all') - par_rheo(1)) / 1e+3;
    pstd(j) = std(samples_total,0,'all') / 1e+3;
    band = quantile(samples_total,[.025,.975],'all');
    coverage(j) = (band(1) < par_rheo(1)) & (band(2) > par_rheo(1));
end
tbl = round([fitted_param(1,:)' bias' pstd']); % Rounded to kPa scale
cat(tbl)