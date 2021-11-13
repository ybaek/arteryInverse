function [eta,wv] = backtrack_ess(l,delta,thres,rho)
%% Similar to well-known backtracking for line search
%% Criterion is effective sample size > thres 
eta = 1;
eff = -Inf;
while (eff < 0)
    wv = exp(-eta*delta*l);
    eff = sum(wv)^2 / sum(wv.^2) - thres;
    if (isnan(eff)) %Inf/Inf situation
        eff = -Inf;
    end
    eta = rho*eta;
end
wv = exp(-eta*delta*l);
return