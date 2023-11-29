clc; clear; close all;
% This script file will run the forward model for a given modulus
% parameter, G0.
% The modulus parameter is defined as G=G0*(i\omega)^\alpha (spring-pot model)

par0=[100e+3;1e-3;3e-3];
alpha=0.16;
freq1=300:10:500;
freq2=900:10:1000;

tic;
cp1=forward_model(par0,alpha,freq1,2);
cp2=forward_model(par0,alpha,freq2,1);
toc;

figure(1);
plot(freq1,cp1,'linewidth',1.75);
hold on
plot(freq2,cp2,'linewidth',1.75);
xlabel('Frequency(Hz)'); ylabel('c_p(m/s)');
ylim([5 10]);
set(gca,'Fontsize',12); set(gca,'Fontweight','bold');


