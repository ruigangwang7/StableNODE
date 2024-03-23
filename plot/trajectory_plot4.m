clear; close all;
% Init
root_dir = pwd; 
StartupPlotting()

% load data
% cd('../results/simu_error_uniform_forfig5/')
cd('../results/simu_error')
icnn = load('icnn_dyn_exp/simu_trajectory.mat');
mlp = load('stable_dyn_exp/simu_trajectory.mat');
ham = load('Ham_dyn_exp/simu_trajectory.mat');
% pdicnn = load('posedef_icnn_dyn_exp/simu_trajectory.mat');

% Get tm for plot
h=mlp.timestep; 
N=mlp.steps;
tm = linspace(h,double(h*N) ,N);
batches = [11 10];
plot_odds = [1 3 5 7];

%% Plot the left half from batches(1)
f = figure;
set(gcf,'Units','inches');
set(gcf, 'Position',  [0 0 16 10.27])
screenposition = get(gcf,'Position');
papersize = screenposition(3:4);
set(gcf,...
    'PaperPosition',[0 0 papersize],...
    'PaperSize',papersize);
i = 1;
% plot theta1
batch_index = batches(i);
state_index = 1;
subplot(4, 2, plot_odds(state_index));
shnd_p = plot(tm, ham.x_model(:, batch_index, state_index), DisplayName='SHND'); hold 
icnn_p = plot(tm, icnn.x_model(:, batch_index, state_index), DisplayName='SD-ICNN'); 
mlp_p = plot(tm, mlp.x_model(:, batch_index, state_index), DisplayName='SD-MLP'); 
true_p = plot(tm, mlp.x_true(:, batch_index, state_index), '--', DisplayName='True'); 
% pdicnn_p = plot(tm, pdicnn.x_model(:, batch_index, state_index), DisplayName='SD-PDICNN'); 
ylabel('$$\theta_1$$')
legend([shnd_p, icnn_p, mlp_p, true_p])%pdicnn_p
title("$x(0) = [-0.4798, 0.6202, 0.0000, 0.0000]^{\top}$")
grid
set(gca, "Position", [0.0511   0.7099    0.4420    0.1750])
ylim([-0.6, 3])

% plot theta22900
state_index = 2;
subplot(4, 2, plot_odds(state_index));
shnd_p = plot(tm, ham.x_model(:, batch_index, state_index), DisplayName='SHND');  hold 
icnn_p = plot(tm, icnn.x_model(:, batch_index, state_index), DisplayName='SD-ICNN'); 
mlp_p = plot(tm, mlp.x_model(:, batch_index, state_index), DisplayName='SD-MLP'); 
true_p = plot(tm, mlp.x_true(:, batch_index, state_index), '--', DisplayName='True'); 
ylabel('$$\theta_2$$')
grid
set(gca, "Position", [0.0511    0.4999    0.4420    0.1750])
ylim([-0.5, 3/2])

% plot dtheta1
state_index = 3;
subplot(4, 2, plot_odds(state_index));
shnd_p = plot(tm, ham.x_model(:, batch_index, state_index), DisplayName='SHND');  hold  
icnn_p = plot(tm, icnn.x_model(:, batch_index, state_index), DisplayName='ICNN'); 
mlp_p = plot(tm, mlp.x_model(:, batch_index, state_index), DisplayName='MLP'); 
true_p = plot(tm, mlp.x_true(:, batch_index, state_index), '--', DisplayName='True'); 
ylabel('$$\dot{\theta}_1$$')
grid
set(gca, "Position", [0.051100    0.2900    0.4420    0.1750])
ylim([-6, 6])

% plot dtheta2
state_index = 4;
subplot(4, 2, plot_odds(state_index));
shnd_p = plot(tm, ham.x_model(:, batch_index, state_index), DisplayName='SHND'); hold 
icnn_p = plot(tm, icnn.x_model(:, batch_index, state_index), DisplayName='ICNN'); 
mlp_p = plot(tm, mlp.x_model(:, batch_index, state_index), DisplayName='MLP'); 
true_p = plot(tm, mlp.x_true(:, batch_index, state_index), '--', DisplayName='True'); 
ylabel('$$\dot{\theta}_2$$')
xlabel('Time (s)')
grid
set(gca, "Position", [0.051100    0.0810    0.4420    0.1750])
ylim([-10, 5])

plot_odds = plot_odds + 1;

%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the right half from batches(2)
i = 2;
% plot theta1
batch_index = batches(i);
state_index = 1;
subplot(4, 2, plot_odds(state_index));
shnd_p = plot(tm, ham.x_model(:, batch_index, state_index), DisplayName='SHND'); hold 
icnn_p = plot(tm, icnn.x_model(:, batch_index, state_index), DisplayName='SD-ICNN'); 
mlp_p = plot(tm, mlp.x_model(:, batch_index, state_index), DisplayName='SD-MLP'); 
true_p = plot(tm, mlp.x_true(:, batch_index, state_index), '--', DisplayName='True'); 
legend([shnd_p, icnn_p, mlp_p, true_p])
title("$x(0) = [-0.4818, 0.6182, 0.0000, 0.0000]^{\top}$")
grid
set(gca, "Position", [0.53   0.7099    0.4420    0.1750])
ylim([-0.6, 3])

% plot theta2
state_index = 2;
subplot(4, 2, plot_odds(state_index));
shnd_p = plot(tm, ham.x_model(:, batch_index, state_index), DisplayName='SHND');  hold 
icnn_p = plot(tm, icnn.x_model(:, batch_index, state_index), DisplayName='ICNN'); 
mlp_p = plot(tm, mlp.x_model(:, batch_index, state_index), DisplayName='MLP'); 
true_p = plot(tm, mlp.x_true(:, batch_index, state_index), '--', DisplayName='True'); 
grid
set(gca, "Position", [0.53    0.4999    0.4420    0.1750])
ylim([-0.5, 3/2])

% plot dtheta1
state_index = 3;
subplot(4, 2, plot_odds(state_index));
shnd_p = plot(tm, ham.x_model(:, batch_index, state_index), DisplayName='SHND');  hold  
icnn_p = plot(tm, icnn.x_model(:, batch_index, state_index), DisplayName='ICNN'); 
mlp_p = plot(tm, mlp.x_model(:, batch_index, state_index), DisplayName='MLP'); 
true_p = plot(tm, mlp.x_true(:, batch_index, state_index), '--', DisplayName='True'); 
grid
set(gca, "Position", [0.53    0.2900    0.4420    0.1750])
ylim([-6, 6])

% plot dtheta2
state_index = 4;
subplot(4, 2, plot_odds(state_index));
shnd_p = plot(tm, ham.x_model(:, batch_index, state_index), DisplayName='SHND'); hold 
icnn_p = plot(tm, icnn.x_model(:, batch_index, state_index), DisplayName='ICNN'); 
mlp_p = plot(tm, mlp.x_model(:, batch_index, state_index), DisplayName='MLP'); 
true_p = plot(tm, mlp.x_true(:, batch_index, state_index), '--', DisplayName='True'); 
xlabel('Time (s)')
grid
set(gca, "Position", [0.53    0.0810    0.4420    0.1750])
ylim([-10, 5])

% some post process to line width etc
cd(root_dir)
FormatNice()
save_name = sprintf('theta12_case12'); 
save_name = fullfile(root_dir,save_name);
% saveas(f, save_name,'pdf')
