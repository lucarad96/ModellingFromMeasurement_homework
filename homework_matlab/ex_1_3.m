% ModellingFromMeasurement
% Homework exercise 1.3

% Hare&Lynx - Lotka-Volterra model:
%
%

clear; close all; clc;

rng(160)

t0 = 1845; % Initial year for the observation
t1 = 1903; % Final yar for the observation
t_span = [t0 t1];
dt = 2; % Time intervals (2 years)
t = t0:dt:t1;

spline_flag = 1;

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

xy = [hare; lynx];
xd = diff(hare);
yd = diff(lynx);
xy_dot = [xd;yd];

%init_opt = [0.5 0.5 0.5 0.5];
%% optmization problem witout ode (with f min unconstrained)

rand_vect = -50:50;
options = optimset('MaxFunEvals',1000,'TolFun',0.01);

obj_f_opt = inf;
for ii = 1:10
    %par_in = rand(1,4);
    %par_in = [0 1 -1 0];
    par_in = randsample(rand_vect,4);

    g = @(par) compute_loss_diff(par,t,xy,xy_dot);
    [obj_par,obj_f] = fminunc(g,par_in,options);

    if obj_f < obj_f_opt
        B = obj_par;
        %par_init = par_in;
        obj_f_opt = obj_f;
    end
end


[tt,xy_pred] = ode23(@lot_volt,t_span,[xy(1,1),xy(2,1)],[],B);

figure
plot(t,hare,'r',t,lynx,'b'); hold on; grid on;
plot(tt,xy_pred(:,1),'r--',tt,xy_pred(:,2),'b--');
ylabel("population, thousands")
xlabel("time, years")
legend('real hare','real linx','model hare','model linx')

%% optmization problem with ode (with f min unconstrained)

%[A,obj_f] = fminsearch(@compute_loss_ode,init_opt,[],t,xy);
% options = optimset('MaxFunEvals',1000,'TolFun',0.01,'Display','iter');
% 
% obj_f_optm = inf;
% for ii = 1:10
%     %par = rand(1,4);
%     %par = [0.5 -1 0 0.5];
%     par_in = randsample(rand_vect,4);
%     disp(['iteration ' num2str(ii)])
%     f = @(par) compute_loss_ode(par,t,xy);
%     %[obj_par,obj_f] = fminunc(@compute_loss_ode,par_in,options,t,xy);
%     [obj_par,obj_f] = fminunc(f,par_in,options);
% 
%     if obj_f < obj_f_optm
%         A = obj_par;
%         %par_init = par_in;
%         obj_f_opt = obj_f;
%     end
% end
% 
% % Visualize Data
% 
% % initial conditions
% 
% [tt,x_solved] = ode23(@lot_volt,t,[xy(1,1),xy(2,1)],[],A);
% 
% figure
% plot(t,hare,'r',t,lynx,'b'); hold on
% plot(tt,x_solved(:,1),'r--',tt,x_solved(:,2),'b--');
% legend('real hare','real linx','model hare','model linx')

%% functions definitions

function final_loss = compute_loss_diff(par,t,xy,xy_dot)

[~,nc] = size(xy);
loss = 0;
% Compute MAE for every datapoints
for ii=1:nc-1
    errors = lot_volt(t,[xy(1,ii),xy(2,ii)],par);
    loss = loss + sum(abs(errors - xy_dot(:,ii)));
end
final_loss = loss;
end

function error = compute_loss_ode(par,t,xy)

% ODE function

[~,out] = ode23(@lot_volt,t,[xy(1,1),xy(2,1)],[],par);

% error calculus
errx = out(:,1)-xy(1,:)';
erry = out(:,2)-xy(2,:)';
error = errx'*errx + erry'*erry;
end

% lotke-Volterra Law
function xy_dot = lot_volt(t,xy,par) % tilde placeolder per t
xy_dot = [(par(1)-par(2)*xy(2))*xy(1);(par(3)*xy(1)-par(4))*xy(2)];
end