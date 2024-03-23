clearvars
close all
clc

root_dir = pwd; 
StartupPlotting()


trainsizes = [4 6 8 10 12 14 16 18 20]*100;

Tloss = [0.06429	0.1618	0.1679
        0.0493	0.2349	0.1267
        0.04412	0.3641	0.7766
        0.03809	0.3267	0.7164
        0.03588	0.2048	0.3313
        0.0331	0.2111	0.3184
        0.02576	0.08513	0.624
        0.02763	0.09573	0.2274
        0.02726	0.09015	0.2644
]; 

Vloss = [4.6911	4.5965	5.0812
        1.5765	1.9267	1.9424
        0.8551	1.7527	2.3267
        0.5297	1.227	1.5778
        0.4428	1.9952	0.7341
        0.3281	0.7851	0.6867
        0.2459	0.4073	0.8332
        0.2038	0.4655	0.6148
        0.1801	0.3998	0.524
        ];

yrange = [0.01 100];
xrange = [400 2000];
%% Plot
f = figure
set(gcf,'Units','inches');
set(gcf, 'Position',  [0 0 8 3.8])
screenposition = get(gcf,'Position');
papersize = screenposition(3:4);
set(gcf,...
    'PaperPosition',[0 0 papersize],...
    'PaperSize',papersize);

subplot(1, 2, 1)
semilogy(trainsizes, Tloss, 'o-')
title("Train")
legend("SHND", "SD-ICNN", "SD-MLP")
ylim(yrange)
xlim(xrange)
ylabel('$$\ell_2$$ loss')
xlabel('Training data size')
grid
set(gca, "Position", [0.1 0.1559 0.40 0.7510])
xticks([400 800 1200 1600 2000]);

subplot(1, 2, 2)
semilogy(trainsizes, Vloss, 'o-')
set(gca, "Position", [0.57 0.1559 0.40 0.7510])
title("Test")
legend("SHND", "SD-ICNN", "SD-MLP")
xlim(xrange)
ylim(yrange)
xlabel('Training data size')
xticks([400 800 1200 1600 2000]);
grid
% some post process to line width etc
cd(root_dir)
FormatNice()
% saveas(f, 'loss_v_trainsize','pdf')
