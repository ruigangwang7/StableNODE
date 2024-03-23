clearvars
close all
clc

root_dir = pwd; 
StartupPlotting()

mu = [0.1 1 3 5 7 10]/10.0;
loss_mu = [0.025 0.175
            0.027	0.180
            0.031	0.159
            0.103	0.197
            0.346	0.414
            1.238	1.245];

nu = [0.25 0.5 1 2 4 8 15];
nu = nu/0.1; 
loss_nu = [0.773	1.315
0.093	0.347
0.033	0.217
0.025	0.175
0.025	0.188	
0.042	0.224
0.094	0.294
]; 
yrange = [0.01 10];

f = figure;
set(gcf,'Units','inches');
set(gcf, 'Position',  [0 0 5 3])
screenposition = get(gcf,'Position');
papersize = screenposition(3:4);
set(gcf,...
    'PaperPosition',[0 0 papersize],...
    'PaperSize',papersize);
% subplot(1,2,1)
% semilogy(mu, loss_mu(:, 1),'-o' , mu, loss_mu(:, 2), 'o-')
% legend('Training loss','Test loss', 'Location','northeast')
% title("Tuning $\mu$")
% ylim(yrange)
% ylabel('$$\ell_2$$ loss')
% xlabel('$\mu$ values while $\nu = 2.0$')
% set(gca, "Position", [0.1 0.1559 0.40 0.7510])
% grid
% 
% subplot(1,2,2)
loglog(nu, loss_nu(:, 1),'-o' , nu, loss_nu(:, 2), 'o-')
legend('Train','Test', 'Location','northeast')
% title("Tuning $\nu$")
ylim(yrange)
ylabel('$$\ell_2$$ loss')
xlabel('${\nu}/{\mu}$ ratio')
xlim([2, 200])
xticks(nu)
grid
% set(gca, "Position", [0.57 0.1559 0.40 0.7510])
% some post process to line width etc
cd(root_dir)
FormatNice()