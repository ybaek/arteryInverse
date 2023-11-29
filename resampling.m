function [inds] = resampling(S,w,method)
% Supports four methods in Chapter 9, Chopin & Papaspiliopoulos
inds = zeros(1,S);
if method == "residual"
    res_w = S * w / sum(w);
    fl_w = floor(res_w);
    frac_w = res_w - fl_w;
    r = int16(sum(frac_w)); % Always will be an integer
    ctr = 1;
    for s=1:S
        if fl_w(s) > 0
            inds(ctr:(ctr+fl_w(s)-1)) = s * ones(1,fl_w(s));
        end
        ctr = ctr + fl_w(s);
    end
    inds(ctr:S) = randsample(S,r,true,frac_w);        
elseif method == "stratified"
    u = zeros(1,S);
    cum_w = cumsum(w)/sum(w);
    for s=1:S
        u(s) = (rand+s-1)/S;
    end
    ctr = 1;
    for s=1:S
        while cum_w(ctr) < u(s)
            ctr = ctr + 1;
        end
        inds(s) = ctr;
    end
elseif method == "systematic"
    u = zeros(1,S);
    u_seed = rand;
    cum_w = cumsum(w)/sum(w);
    for s=1:S
        u(s) = (u_seed+s-1)/S;
    end
    ctr = 1;
    for s=1:S
        while cum_w(ctr) < u(s)
            ctr = ctr + 1;
        end
        inds(s) = ctr;
    end
else
    % If anything else, resort to multinomial sampler
    inds = randsample(S,S,true,w)';
end
return
