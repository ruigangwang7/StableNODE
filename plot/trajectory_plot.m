clearvars
close all
clc
% 
root_dir = pwd; 
StartupPlotting()

cd('../results/simu_error/')
%%
files = dir; 

x_physics = cell(3,1);
x_modelp = cell(3,1); 

%%
%Load and nominate data
for i = 3:length(files)
    clear x_true x_model t
    
    file = fullfile(files(i).name,'simu_trajectory.mat');
    load(file)
    
    x_physics{i-2} = x_true;
    x_modelp{i-2} = x_model; 
    files(i).name
%     mean_error
end

t = linspace(timestep,double(timestep*steps) ,steps); 
%%


for i = 1
    close all
    f = figure; 
    
%     set(gcf,'Units','inches');
%     set(gcf, 'Position',  [7.2708 3.1771 11.4896 5.0625])
%     screenposition = get(gcf,'Position');
%     papersize = screenposition(3:4);
%     set(gcf,...
%         'PaperPosition',[0 0 papersize],...
%         'PaperSize',[papersize]);
    
    batch_index = i;
    state_index = 1; 
%     batch_index = 108;

    % Plot theta1
    subplot(1,2,1)
    plot(t, x_modelp{1}(:, batch_index, state_index), ...
        t, x_modelp{2}(:, batch_index, state_index), ...
        t, x_modelp{3}(:, batch_index, state_index), ...
        t, x_physics{1}(:, batch_index, state_index),'-k')
    title('$$\theta_1$$')
    ylabel('Pendulum angle (rad)')
    xlabel('Time (s)')
    legend('SHND', 'ICNN', 'MLP', 'Physical')
    set(gca, "Position", [0.1 0.1299 0.40 0.8150])
    grid
    
    % Plot theta2
    state_index = 2; 
    subplot(1,2,2)
    plot(t, x_modelp{1}(:, batch_index, state_index), ...
        t, x_modelp{2}(:, batch_index, state_index), ...
        t, x_modelp{3}(:, batch_index, state_index), ...
        t, x_physics{1}(:, batch_index, state_index),'-k')
    title('$$\theta_2$$')
    xlabel('Time (s)')
    legend('SHND', 'ICNN', 'MLP', 'Physical')
    grid
%     set(gca, "Position", [0.55 0.1299 0.40 0.8150])
%     i;

%     save for ray to check
%     save_name = sprintf('batch_%i', i); 
%     saveas(gcf, save_name,'png')
    
%     % some post process to line width etc
%     cd(root_dir)
%     FormatNice()
%     % Formal save
%     save_name = sprintf('theta12_batch_%i', batch_index); 
%     save_name = fullfile(root_dir,save_name);
%     saveas(f, save_name,'pdf')
end