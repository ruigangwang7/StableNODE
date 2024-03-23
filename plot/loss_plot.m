clearvars
close all

root_dir = pwd; 
StartupPlotting()

% cd('../results/train_size/pendulum2_trainsize1800/')
cd('../results//pendulum2/')

files = dir; 

Tloss = cell(3,1);
Vloss = cell(3,1); 

for i = 3:length(files)
    clear vloss tloss
    file = fullfile(files(i).name,'loss.mat');
    load(file)

    Tloss{i-2} = tloss;
    Vloss{i-2} = vloss; 
end

Epoch = 1:length(Tloss{1}); 
yrange = [0.01 100];
%%
f = figure; 
set(gcf,'Units','inches');
set(gcf, 'Position',  [0 0 8 3.68])
% [4.5833 3.4375 10.9375 4.8333]
screenposition = get(gcf,'Position');
papersize = screenposition(3:4);
set(gcf,...
    'PaperPosition',[0 0 papersize],...
    'PaperSize',papersize);

%%
subplot(1,2,1)
semilogy(Epoch, Tloss{1}, Epoch, Tloss{2}, Epoch, Tloss{3})
title("Train")
legend("SHND", "SD-ICNN", "SD-MLP")
ylim(yrange)
ylabel('$$\ell_2$$ loss')
xlabel('Epoch')
grid
set(gca, "Position", [0.1 0.1559 0.40 0.7510])

subplot(1,2,2)
semilogy(Epoch, Vloss{1}, Epoch, Vloss{2}, Epoch, Vloss{3})
set(gca, "Position", [0.57 0.1559 0.40 0.7510])
title("Test")
legend("SHND", "SD-ICNN", "SD-MLP")
ylim(yrange)
% ylabel('$$\ell^2$$ loss')
xlabel('Epoch')
grid
% Post processing
cd(root_dir)
FormatNice(16,1.2)

save_name = "loss_v_epoch.pdf";
save_name = fullfile(root_dir,save_name);


% saveas(f, save_name,'pdf')
