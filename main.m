%% Bayesian inverse problem using Tuhin's code
clc; clear; close all; 
rng('default'); seed = rng;
delete(gcp('nocreate'));
parpool(4);
%% Simulating forward model & noisy data
G0=100e+3; 
[cp,freq]=forward_model(G0);
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
lb = 50e+3;
ub = 150e+3;
loss = @(y,g0) sum((y - forward_model(g0)).^2); % loss per each datum
lprior = @(g0) -betalike([2,2],(g0-lb)/(ub-lb)) +...
    log((ub-g0)*(g0-lb)/(ub-lb)); % logit-scale Jacobian
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
mean_loo = zeros(1,N);
cov_term = zeros(1,N);
obj_term = zeros(1,N);
deriv_term = zeros(1,N);
%% Computation
tic
while (t < Tmax)
    t = t + 1;
    if (~t)
        G0_draws = 100e+3*betarnd(2,2,[1,S])+50e+3;
        W = Wmin;
        parfor s=1:S
            % Step 1: Reweighting
            % USES FORWARD_MODEL
            wv(s) = exp(-W*sum(loss(noisy_cp,G0_draws(s))));
        end
        continue
    else
        % Step 2: Resampling
        % USES RNG
        ancstr = resampling(S,wv,"stratified");
        G0_draws = G0_draws(:,ancstr);
        % Step 3: Mutation
        Tdraws = log(G0_draws-lb)-log(ub-G0_draws);
        Sigma = 2.38^2*(...
            sum(wv.*(Tdraws.^2))/sum(wv) - sum(wv.*Tdraws)^2/sum(wv)^2);
        parfor s=1:S
            for k=1:K
                % Metropolis-Hastings kernel
                % USES FORWARD_MODEL
                l = loss(noisy_cp,G0_draws(s));
                ll_x = -W*sum(l);
                ll_x = ll_x + lprior(G0_draws(s));
                % USES RNG
                xstar_T = normrnd(Tdraws(s),Sigma);
                xstar = lb./(1+exp(xstar_T)) + ub./(1+exp(-xstar_T));
                % USES FORWARD_MODEL
                lstar = loss(noisy_cp,xstar);
                ll_xstar = -W*sum(lstar);
                ll_xstar = ll_xstar + lprior(xstar);
                log_r = ll_xstar - ll_x;
                log_alpha = min(log_r, 1);
                coin = rand;
                if (log(coin)<log_alpha)
                    G0_draws(s) = xstar;
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
            mean_loo(i) = sum(wv_loo .* G0_draws) / sum(wv_loo);
            loss_loo = wv_loo .* loss_mat(loo_inds,:) / sum(wv_loo);
            meanloss_loo = sum(loss_loo,'all');
            meanprod_loo = sum(loss_loo .* G0_draws,'all');
            cov_term(i) = meanprod_loo - meanloss_loo*mean_loo(i);
        end
        parfor i=1:N
            forward_mean = forward_model(mean_loo(i));
            li = forward_mean - noisy_cp(:,i);
            obj_term(i) = sum(li.^2);
            % Crude approx of gradient of the forward_model
            h = .1e+3;
            fd = (forward_model(mean_loo(i)+h) - forward_mean) / h; 
            deriv_term(i) = -2/N*cov_term(i) * (li'*fd);
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