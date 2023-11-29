function [samples,loss_mat,W,rsfb,rsfb_se,log_wv] = smc_sfb(dat,P,S,K,emin,grid,prior_opt,forward_opt)
T = length(grid);
[~,N] = size(dat);
hp = prior_opt.hp;
lb = prior_opt.lb;
ub = prior_opt.ub;
samples = zeros(P,S,T+1);
loss_mat = zeros(N,S,T+1);
log_wv = zeros(S,T);
rsfb = zeros(1,T);
rsfb_se = zeros(1,T);
% 0th step: Generate particles from the prior
samples(:,:,1) = prior_rng(hp,lb,ub,[P,S]);
fprintf('Initializing samples and weights...\n');
parfor s=1:S
    loss_mat(:,s,1) = Loss(dat,samples(:,s,1),forward_opt);
end
W0 = 0;
fprintf('Iteration \t ESS\n');
for t = 2:(T+1)
    W = grid(t-1);
    log_wv(:,t-1) = -(W-W0)*sum(loss_mat(:,:,t-1)) +...
        logsumexp(W*loss_mat(:,:,t-1)) - logsumexp(W0*loss_mat(:,:,t-1));
    l_ess = 2*logsumexp(log_wv(:,t-1)) - logsumexp(2*log_wv(:,t-1));
    fprintf('  %d  \t  %.1g\n', t, exp(l_ess));
    if (exp(l_ess) < emin)
        fprintf('ESS fell below %.1g , resample and mutate\n', emin);
        % Resample
        % Decide ancestors
        ancstr = resampling(S,exp(log_wv(:,t-1)),"stratified");
        samples(:,:,t) = samples(:,ancstr,t-1);
        loss_mat(:,:,t) = loss_mat(:,ancstr,t-1);
        log_wv(:,t-1) = log_wv(ancstr,t-1);
        wv_n = exp(log_wv(:,t-1) - logsumexp(log_wv(:,t-1)));
        % Mutate
        % Particle-adaptive M-H proposal kernel
        Tsamples = Tmap(samples(:,:,t),lb,ub);
        wcov = (Tsamples .* wv_n') * Tsamples';
        wmean = sum(Tsamples .* wv_n', 2);
        wvar = (trace(wcov) - wmean'*wmean) / P;
        Sigma = 2.38^2 /P * (wcov - wmean*wmean' + wvar*eye(P));
        Sigma = (Sigma + Sigma') / 2;
        parfor s=1:S
            old = samples(:,s,t);
            l_old = loss_mat(:,s,t);
            for k=1:K
                new = old;
                l_new = l_old;
                nll1 = W*sum(l_new) - logsumexp(W*l_new);
                prop_T = mvnrnd(Tmap(old,lb,ub),Sigma)';
                prop = Tinv(prop_T,lb,ub);
                l_prop = Loss(dat,prop,forward_opt);
                nll2 = W*sum(l_prop) - logsumexp(W*l_prop);
                log_r = nll1-nll2;
                log_r = log_r +...
                    (prior_log_density(prop,hp,lb,ub) - prior_log_density(new,hp,lb,ub));
                log_r = log_r + (Tjacobian(prop,lb,ub) - Tjacobian(new,lb,ub));
                coin = rand;
                if (log(coin) < log_r)
                    new = prop;
                    l_new = l_prop;
                end
            end
            samples(:,s,t) = new;
            loss_mat(:,s,t) = l_new;
        end
        log_wv(:,t-1) = zeros(S,1);
    else
        samples(:,:,t) = samples(:,:,t-1);
        loss_mat(:,:,t) = loss_mat(:,:,t-1);
    end
    W0 = W;
    [r,se] = szmix(W,loss_mat(:,:,t),log_wv(:,t-1)');
    rsfb(t-1) = r; rsfb_se(t-1) = se;
end
[rm,ind] = min(rsfb);
ind = find(rsfb < rm + rsfb_se(ind), 1, 'first');
W = grid(ind);
samples = samples(:,:,ind+1);
loss_mat = loss_mat(:,:,ind+1);
log_wv = log_wv(:,ind)';

% ind = T;
% W = grid(ind);
% samples = samples(:,:,ind+1);
% loss_mat = loss_mat(:,:,ind+1);
% log_wv = log_wv(:,ind);
end