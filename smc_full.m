function [samples_f,loss_mat_f] = smc_full(samples_p,W,loss_mat_p,dat,K,prior_opt,forward_opt)
[P,S] = size(samples_p);
hp = prior_opt.hp;
lb = prior_opt.lb;
ub = prior_opt.ub;
% Weight
log_wv = -logsumexp(W*loss_mat_p);
% Resample
ancstr = resampling(S,exp(log_wv),"stratified");
samples_f = samples_p(:,ancstr);
loss_mat_f = loss_mat_p(:,ancstr);
log_wv = log_wv(ancstr);
wv_n = exp(log_wv - logsumexp(log_wv));
% Mutate: particle-adaptive cov kernel
Tsamples = Tmap(samples_p,lb,ub);
wcov = (Tsamples .* wv_n) * Tsamples';
wmean = sum(Tsamples .* wv_n, 2);
wvar = trace(wcov) - wmean'*wmean;
Sigma = 2.38^2 /P * (wcov - wmean*wmean' + wvar*eye(P));
parfor s=1:S
    old = samples_f(:,s);
    l_old = loss_mat_f(:,s);
    for k=1:K
        new = old;
        l_new = l_old;
        nll1 = W*sum(l_new);
        prop_T = mvnrnd(Tmap(old,lb,ub),Sigma)';
        prop = Tinv(prop_T,lb,ub);
        l_prop = Loss(dat,prop,forward_opt);
        nll2 = W*sum(l_prop);
        log_r = nll1-nll2;
        log_r = log_r + (prior_log_density(prop,hp,lb,ub) - prior_log_density(new,hp,lb,ub));
        log_r = log_r + (Tjacobian(prop,lb,ub) - Tjacobian(new,lb,ub));
        coin = rand;
        if (log(coin) < log_r)
            new = prop;
            l_new = l_prop;
        end
    end
    samples_f(:,s) = new;
    loss_mat_f(:,s) = l_new;
end
end