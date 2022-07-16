% ModellingFromMeasurement 
% Homework exercise 1.1

% Hare&Lynx - First analysis:
% - Exact DMD
% - Optimal DMD
% - Bagging with exact DMD
% - Bagging with optimal DMD

% GROSSO DUBBIO SUL BAGGING. 
% Da come ci aveva spiegato il Nathan, possiamo
% ottenere una distribuzione dei parametri b,Phi,lambda. E utilizziamo la
% formula per predire avendo come parametri delle distribuzioni... non è
% proprio quello che abbiamo fatto qua; qua abbiamo addestrato tot modelli,
% fatto una previsione per ogni modello, e infine estratto media e
% varianza.

clear; close all; clc;

%set rangen
rng(2)

spline_flag = 1; %CAMBIARE SE NON SI VUOLE UTILIZZARE LA SPLINE

t0 = 1845; % Initial year for the observation
t1 = 1903; % Final yar for the observation
dt = 2; % Time intervals (2 years)
t = t0:dt:t1;

hare =  [20, 20, 52, 83, 64, 68, 83, 12, 36, 150, 110, 60, 7, 10, 70, 100, 92, 70, 10, 11, 137, 137, 18, 22, 52, 83, 18, 10, 9, 65];
lynx =  [32, 50, 12, 10, 13, 36, 15, 12, 6, 6, 65, 70, 40, 9, 20, 34, 45, 40, 15, 15, 60, 80, 26, 18, 37, 50, 35, 12, 12, 25];


if spline_flag
    % If spline flag == 1, verranno utilizzte delle spline per interpolare
    % i punti
    spline_dt = 0.5;
    spline_t = t0:spline_dt:t1;
    
    figure
    s_hare = spline(t,hare,spline_t);
    plot(t,hare,'ro',spline_t,s_hare,'r'); hold on; grid on;
    s_lynx = spline(t,lynx,spline_t);
    plot(t,lynx,'bo',spline_t,s_lynx,'b')
    ylabel("population, thousands")
    xlabel("time, years")
    legend("hare","hare spline","lynx","lynx spline")
    
    % new names:
    hare = s_hare;
    lynx = s_lynx;
    t = spline_t;
    dt = spline_dt;

end

time_series = [hare; lynx];
[~,n_snapshots] = size(time_series); % n_snap = 30 time intervals

%% Exact DmD

[x_dmd,eigenval] = exact_DMD(time_series,t,time_series(:,1));

figure
plot(eigenval,'*'); grid on; axis([0.5 2.5 0 1])

figure
plot(t,hare,'r'); hold on; grid on;
plot(t,lynx,'b'); 
plot(t,x_dmd(1,:),'r*');
plot(t,x_dmd(2,:),'b*');
ylabel("population, thousands")
xlabel("time, years")
legend('hare real','linx real','hare dmd','lynx dmd')

%% Optimal DmD

% Dimension to retain
r = 2;

% Linear constraints for the optizer in optdmd.
% We here specify that the eigenvalues must have only negative-real part
% while every value for the imaginary part is allowed.
low_b = [-Inf*ones(r,1); -Inf*ones(r,1)];
upp_b = [zeros(r,1); Inf*ones(r,1)];

copts = varpro_lsqlinopts('lbc',low_b,'ubc',upp_b);
opts = []; % varpro_opts('maxiter',1000);

r = 2;
[w,e,b] = optdmd(time_series,t,r,1,opts,[],[],copts);
x_opt_dmd = w*diag(b)*exp(e*t); 

plot(t,real(x_opt_dmd(1,:)),'ro'); 
plot(t,real(x_opt_dmd(2,:)),'bo'); 
legend('hare real','linx real','hare dmd','lynx dmd','hare optDmd','lynx optDmd')



%% Bagging input generator

% https://www.ibm.com/cloud/learn/bagging
% According this resource, in a bagging a sample can be taken more times.

% 2 Hyperpar: 
n_samples = 30; % must be less than 30 % or 117 if spline...
n_learners = 200; % NUmber of models that will be built

% Random indices to create a bagging model
idx_bag_mat = zeros(n_learners,n_samples);
for ii = 1:n_learners
    rand_vector = sort(randsample(1:n_snapshots,n_samples));
    idx_bag_mat(ii,:) = rand_vector;
end

%% Exact dmd with bagging ================================================

figure
bag_ex_dmd.hare = zeros(n_learners,n_snapshots);
bag_ex_dmd.lynx = zeros(n_learners,n_snapshots);
for ii = 1:n_learners
    % Extraction of the random (but sorted) indices
    idx_bag = idx_bag_mat(ii,:);

    % Extraction of the data matrix and time vector
    X = time_series(:,idx_bag);

    % Exact DMD
    [x_dmd,~] = exact_DMD(X,t,time_series(:,1));
    bag_ex_dmd.hare(ii,:) = real(x_dmd(1,:));
    bag_ex_dmd.lynx(ii,:) = real(x_dmd(2,:));

    % Plot
    plot(t,real(bag_ex_dmd.hare(ii,:)),'color',[0.9 0.7 0.7],'LineWidth',0.52); hold on;
    plot(t,real(bag_ex_dmd.lynx(ii,:)),'color',[0.7 0.7 0.9],'LineWidth',0.2); grid on;
    clear x_dmd
end

% Computation of mean and STD
bag_ex_dmd.hare_mean = mean(bag_ex_dmd.hare);
bag_ex_dmd.hare_std = std(bag_ex_dmd.hare);
bag_ex_dmd.lynx_mean = mean(bag_ex_dmd.lynx);
bag_ex_dmd.lynx_std = std(bag_ex_dmd.lynx);

% Confidence hare
t_conf = [t t(end:-1:1)];
y = bag_ex_dmd.hare_mean-bag_ex_dmd.hare_std;
hare_conf = [bag_ex_dmd.hare_mean+bag_ex_dmd.hare_std, y(end:-1:1)];
r = fill(t_conf,hare_conf,'blue');
r.FaceColor = [1 0.8 0.8]; 
r.FaceAlpha = 0.7;
r.EdgeColor = 'none';  

% Confidence lynx
y = bag_ex_dmd.lynx_mean-bag_ex_dmd.lynx_std;
lynx_conf = [bag_ex_dmd.lynx_mean+bag_ex_dmd.lynx_std, y(end:-1:1)];
b = fill(t_conf,lynx_conf,'blue');
b.FaceColor = [0.8 0.8 1]; 
b.FaceAlpha = 0.7;
b.EdgeColor = 'none';  

% Plot means
a = plot(t,bag_ex_dmd.hare_mean,'r','LineWidth',2,'DisplayName','mean hare');
b = plot(t,bag_ex_dmd.lynx_mean,'b','LineWidth',2,'DisplayName','mean lynx');
ylabel("population, thousands")
xlabel("time, years")
legend([a,b])
% come inserisco la legenda per gli utlimi due ingressi?

%% Optimal dmd with bagging ================================================

figure
bag_op_dmd.hare = NaN(n_learners,n_snapshots);
bag_op_dmd.lynx = NaN(n_learners,n_snapshots);
bag_op_dmd.t = NaN(n_learners,n_samples);

%max_w = zeros(1,n_learners);
for ii = 1:n_learners %64
    % Extraction of the random (but sorted) indices
    idx_bag = idx_bag_mat(ii,:);

    % Extraction of the data matrix and time vector
    X = time_series(:,idx_bag);
    t_bag = t(idx_bag);
    bag_op_dmd.t(ii,:) = t_bag;

    % optimal DMD
    [w,e,b] = optdmd(X,t_bag,2,1); %,opts,[],[],copts);

    % We retain only the models with non-positive eigenvalues and
    % non-positive eigenvectors' elements
    if any([real(w) <= 0, real(e) <= 0])

        % Computing next states
        x_opt_dmd = w*diag(b)*exp(e*t);
        bag_op_dmd.hare(ii,:) = real(x_opt_dmd(1,:));
        bag_op_dmd.lynx(ii,:) = real(x_opt_dmd(2,:)); 
        
        % plots
        plot(t,real(bag_op_dmd.hare(ii,:)),'color',[0.9 0.7 0.7],'LineWidth',0.2); hold on;
        plot(t,real(bag_op_dmd.lynx(ii,:)),'color',[0.7 0.7 0.9],'LineWidth',0.2); grid on;
    end
    clear x_dmd
end

% Computation of mean and STD
bag_op_dmd.hare_mean = mean(bag_op_dmd.hare,'omitnan');
bag_op_dmd.hare_std = std(bag_op_dmd.hare,'omitnan');
bag_op_dmd.lynx_mean = mean(bag_op_dmd.lynx,'omitnan');
bag_op_dmd.lynx_std = std(bag_op_dmd.lynx,'omitnan');

% Confidence hare
t_conf = [t t(end:-1:1)];
y = bag_op_dmd.hare_mean-bag_op_dmd.hare_std;
hare_conf = [bag_op_dmd.hare_mean+bag_op_dmd.hare_std, y(end:-1:1)];
r = fill(t_conf,hare_conf,'blue');
r.FaceColor = [1 0.8 0.8]; 
r.FaceAlpha = 0.7;
r.EdgeColor = 'none';  

% Confidence lynx
y = bag_op_dmd.lynx_mean-bag_op_dmd.lynx_std;
lynx_conf = [bag_op_dmd.lynx_mean+bag_op_dmd.lynx_std, y(end:-1:1)];
b = fill(t_conf,lynx_conf,'blue');
b.FaceColor = [0.8 0.8 1]; 
b.FaceAlpha = 0.7;
b.EdgeColor = 'none';  

% Plot means
a = plot(t,bag_op_dmd.hare_mean,'r','LineWidth',2,'DisplayName','mean hare');
b = plot(t,bag_op_dmd.lynx_mean,'b','LineWidth',2,'DisplayName','mean lynx');
ylabel("population, thousands")
xlabel("time, years")
legend([a,b])

