function [result,se] = szmix(W,ls,lw)
log_w_un = W*ls - logsumexp(W*ls) + lw;
log_w_n = log_w_un - logsumexp(log_w_un,2);
result = mean(sum(ls.*exp(log_w_n),2));
se = std(sum(ls.*exp(log_w_n),2));
end