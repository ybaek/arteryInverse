%% Bayesian inverse problem using Tuhin's code
clc; clear; close all; 
rng('default'); seed = rng;
delete(gcp('nocreate'));
parpool(4);
%% Simulating forward model & noisy data
G0=1.1e+5;
thickness=.9e-3;
radius=3.5e-3;
par0=[G0;thickness;radius];
P=3; % no. of parameters
[cp,freq]=forward_model(par0);
figure(1)
plot(freq,cp)

N=30;
D=length(cp);
sigma0=.02;
eps=normrnd(0,sigma0,[D,N]);
noisy_cp = exp(eps) .* cp;
figure(2)
plot(freq, noisy_cp)
%% Convenience anonymous functinos
lb = [50e+3;.5e-3;1.5e-3];
ub = [150e+3;2e-3;4.5e-3];
scalev = ub-lb;
loss = @(y,theta) sum((y - forward_model(theta)).^2);
lprior = @(theta) -betalike([2,2],(theta-lb)./(ub-lb)) +...
    sum(log((ub-theta).*(theta-lb)./(ub-lb)));
%% Computation parameters
S = 400; % no. of particles
Tmax = 10; % maximal steps to proceed
K = 5; % Markov kernel chain length
rho = .9; % line search parameter
Wmin = 5e-4; % Initial hyperparamter

t = -1;
W = 0;
wv = zeros(1,S);
par = zeros(1,Tmax+1);
obj = zeros(1,Tmax+1);
grad = zeros(1,Tmax+1);
loss_mat = zeros(N,S);
mean_loo = zeros(P,N);
cov_term = zeros(P,N);
obj_term = zeros(1,N);
deriv_term = zeros(1,N);
%% Computation
tic
while (t < Tmax)
    t = t + 1;
    if (~t)
        theta_draws = lb + scalev .* betarnd(2,2,[P,S]);
        W = Wmin;
        parfor s=1:S
            % Step 1: Reweighting
            % USES FORWARD_MODEL
            wv(s) = exp(-W*sum(loss(noisy_cp,theta_draws(:,s))));
        end
        continue
    else
        % Step 2: Resampling
        % USES RNG
        ancstr = resampling(S,wv,"stratified");
        theta_draws = theta_draws(:,ancstr);
        % Step 3: Mutation
        Tdraws = log(theta_draws-lb)-log(ub-theta_draws);
        Sigma = 2.38^2/D * (...
            ((Tdraws.*wv)*Tdraws')/sum(wv) -...
            (sum(Tdraws.*wv,2)*sum(Tdraws.*wv,2)')/sum(wv)^2);
        parfor s=1:S
            for k=1:K
                % Metropolis-Hastings kernel
                % USES FORWARD_MODEL
                l = loss(noisy_cp,theta_draws(:,s));
                ll_x = -W*sum(l);
                ll_x = ll_x + lprior(theta_draws(:,s));
                % USES RNG
                xstar_T = mvnrnd(Tdraws(:,s),Sigma)';
                xstar = lb./(1+exp(xstar_T)) + ub./(1+exp(-xstar_T));
                % USES FORWARD_MODEL
                lstar = loss(noisy_cp,xstar);
                ll_xstar = -W*sum(lstar);
                ll_xstar = ll_xstar + lprior(xstar);
                log_r = ll_xstar - ll_x;
                log_alpha = min(log_r, 1);
                coin = rand;
                if (log(coin)<log_alpha)
                    theta_draws(:,s) = xstar;
                    l = lstar;
                end
                if (k==K) % Store these last ones
                    loss_mat(:,s) = l';
                end
            end
        end
        % Compute the needed CV objective and gradient
        % Nested: Another importance sampling
        for i=1:N
            loo_inds = ([1:1:N]~=i);
            wv_loo = exp(W * loss_mat(i,:));
            mean_loo(:,i) = sum(theta_draws.*wv_loo,2) / sum(wv_loo);
            loss_loo = wv_loo.*loss_mat(loo_inds,:) / sum(wv_loo);
            meanloss_loo = sum(loss_loo,'all');
            meanprod_loo = sum(sum(loss_loo).*theta_draws,2);
            cov_term(:,i) = meanprod_loo - meanloss_loo*mean_loo(:,i);
        end
        parfor i=1:N
            forward_mean = forward_model(mean_loo(:,i));
            li = forward_mean - noisy_cp(:,i);
            obj_term(i) = sum(li.^2);
            % Finite step approx of gradient of forward_model
            hv = [1e+5;1e-3;1e-3]; % scales each direction
            eps = 1e-3;
            fdt = zeros(D,P);
            for p=1:P
               tmp = mean_loo(:,i);
               tmp(p) = tmp(p) + eps*hv(p);
               fdt(:,p) = (forward_model(tmp) - forward_mean) / (eps*hv(p));
            end
            deriv_term(i) = -2/N*li'*fdt*cov_term(:,i);
        end
        rcv = sum(obj_term);
        gradcv = sum(deriv_term);
        if (gradcv > 0) % If not valid descent, just stop
            break
        end
        % Re: Step 1: Reweighting w/ adaptive step size
        [eta,wv] = backtrack_ess(sum(loss_mat),-gradcv,S/2,rho);
        W = W - eta*gradcv;
        par(t+1) = W;
        obj(t+1) = rcv;
        grad(t+1) = gradcv;
    end
end
toc

delete(gcp('nocreate'));

%% Added afterwards
post_mean = mean(theta_draws,2);
% Joint posterior has substantial negative correlation
xlabel('G0')
ylabel('thickness')
% Marginal credible interval is unnecessarily wide
% Approx. joint credible ellipsoid can be obtained instead
[U,Sig,V] = svd(theta_draws'-mean(theta_draws,2)','econ');
rot_sc = U ./ std(U); % Approx. Gaussian
sph_dist = sum(rot_sc.^2, 2);
thres = quantile(sph_dist, .95);
cred_inds = find(sph_dist < thres);
cred_set = theta_draws(:,cred_inds);
% "Boundary" of credible set
[sorted,argsort] = sort(sph_dist,'descend');
figure(3)
plot(freq,noisy_cp)
hold on
plot(freq,forward_model(par0),'--','LineWidth',2.5,'Color','r')
for j=1:5
    plot(freq,forward_model(cred_set(argsort(j),:)),...
         '-.','LineWidth',1,'Color','k')
end
hold off