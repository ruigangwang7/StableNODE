%%
clearvars; close all;
% Init
root_dir = pwd; 
StartupPlotting()
cCell = {[0,114,178],[213,94,0],[0,158,115],0.25*255*[1,1,1],...
                     [230,159,0],[86,180,233],[240,215,66],[204,121,167]};
% load data
cd('../results/simu_error/')

icnn = load('icnn_dyn_exp/simu_trajectory.mat');
mlp = load('stable_dyn_exp/simu_trajectory.mat');
ham = load('Ham_dyn_exp/simu_trajectory.mat');
%%
% Get tm for plot
h=mlp.timestep; 
N=mlp.steps;
tm = linspace(h,double(h*N) ,N);

E_mlp = [mlp.mean_error - mlp.min_error, mlp.max_error - mlp.mean_error];
E_ham = [ham.mean_error - ham.min_error, ham.max_error - ham.mean_error];
E_icnn = [icnn.mean_error - icnn.min_error, icnn.max_error - icnn.mean_error];

f = figure;
set(gcf,'Units','inches');
set(gcf, 'Position',  [0 0 5 3])
screenposition = get(gcf,'Position');
papersize = screenposition(3:4);
set(gcf,...
    'PaperPosition',[0 0 papersize],...
    'PaperSize',papersize);

% shnd_p = boundedline(tm, ham.mean_error', E_ham, 'Color', cCell{1}/255, 'alpha'); 
% hold
% icnn_p = boundedline(tm, icnn.mean_error', E_icnn, 'Color', cCell{2}/255, 'alpha'); 
% mlp_p = boundedline(tm, mlp.mean_error', E_mlp, 'Color', cCell{3}/255, 'alpha'); 
shnd_p = plot(tm, ham.mean_error', DisplayName='SHND'); hold 
icnn_p = plot(tm, icnn.mean_error', DisplayName='SD-ICNN'); 
mlp_p = plot(tm, mlp.mean_error', DisplayName='SD-MLP'); 
% icnn_p = plot(tm, icnn.mean_error', DisplayName='SD-ICNN'); 
% mlp_p = plot(tm, mlp.mean_error', DisplayName='SD-MLP'); 
% legend(['SHND', 'ICNN', 'MLP'], 'Location','northeast')
ylabel(' $| \hat{x}(t)-x(t) |$ ')
xlabel('Time (s)')
grid


% some post process to line width etc
cd(root_dir)
FormatNice()
plot(tm, mlp.max_error,'--', 'Color', cCell{3}/255, 'LineWidth',0.8)
plot(tm, icnn.max_error,'--', 'Color', cCell{2}/255, 'LineWidth',0.8)
plot(tm, ham.max_error,'--', 'Color', cCell{1}/255, 'LineWidth',0.8)

legend([shnd_p, icnn_p, mlp_p], 'Location', 'northeast')


% saveas(f, 'mean_simu_error','pdf')
