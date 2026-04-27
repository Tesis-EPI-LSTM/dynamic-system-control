%% ========================================================================
%  MPC vs Baselines – Modelo reducido derivado de LSTM
%  Script listo para Q1
% ========================================================================

clear; close all; clc;
warning('off','mpc:Weights:Empty'); % Ignorar warnings de pesos vacíos

%% ------------------------------------------------------------------------
%% 1) CARGA DEL MODELO REDUCIDO
% -------------------------------------------------------------------------
load('linear_model_optimized_extended.mat');  % A_opt, B_opt, C_total, D_total, Ts

[nx, nu] = size(B_opt);
ny = size(C_total,1);

sys_red = ss(A_opt, B_opt, C_total, D_total, Ts);

%% ------------------------------------------------------------------------
%% 2) MPC ROBUSTO
% -------------------------------------------------------------------------
PredictionHorizon = 30;
ControlHorizon    = 10;

mpc_controller = mpc(sys_red, Ts, PredictionHorizon, ControlHorizon);

% Restricciones
for i = 1:nu
    mpc_controller.MV(i).Min = -3;
    mpc_controller.MV(i).Max =  3;
end
for i = 1:ny
    mpc_controller.OV(i).Min = 0;
    mpc_controller.OV(i).Max = 10;
end

% Pesos
mpc_controller.Weights.OutputVariables          = 50 * ones(ny,2);
mpc_controller.Weights.ManipulatedVariablesRate = 5  * ones(nu,2);
mpc_controller.Weights.ManipulatedVariables     = 0.1* ones(nu,2);
mpc_controller.Weights.ECR = 1e5;

%% ------------------------------------------------------------------------
%% 3) SIMULACIÓN MPC ROBUSTO
% -------------------------------------------------------------------------
N = 100;
t = (0:N-1)'*Ts;

ref = [linspace(1,5,N)', linspace(2,8,N)'];
ref = smoothdata(ref,'movmean',7);

x0 = pinv(C_total)*ref(1,:)';

simopt = mpcsimopt(mpc_controller);
simopt.PlantInitialState = x0;

[y_mpc,~,u_mpc,x_mpc] = sim(mpc_controller,N,ref,simopt);

%% ------------------------------------------------------------------------
%% 4) BASELINE: MPC simple
% -------------------------------------------------------------------------
PredictionHorizon_simple = 10;
ControlHorizon_simple    = 5;

mpc_baseline = mpc(sys_red, Ts, PredictionHorizon_simple, ControlHorizon_simple);

% Restricciones
for i = 1:nu
    mpc_baseline.MV(i).Min = -3;
    mpc_baseline.MV(i).Max = 3;
end
for i = 1:ny
    mpc_baseline.OV(i).Min = 0;
    mpc_baseline.OV(i).Max = 10;
end

% Pesos (ny x 2)
mpc_baseline.Weights.OutputVariables          = 10*ones(ny,2);
mpc_baseline.Weights.ManipulatedVariablesRate = ones(nu,2);
mpc_baseline.Weights.ManipulatedVariables     = 0.01*ones(nu,2);
mpc_baseline.Weights.ECR = 1e5;

simopt_baseline = mpcsimopt(mpc_baseline);
simopt_baseline.PlantInitialState = x0;

[y_baseline,~,u_baseline,x_baseline] = sim(mpc_baseline,N,ref,simopt_baseline);

%% ------------------------------------------------------------------------
%% 5) BASELINE: PID MIMO
% -------------------------------------------------------------------------
% PID paralelo: cada salida con su entrada
Kp = [0.8 0; 0 0.7];   % Proporcional
Ki = [0.1 0; 0 0.05];  % Integral
Kd = [0 0; 0 0];        % Derivativo (opcional)

u_pid = zeros(N, nu);
y_pid = zeros(N, ny);
int_error = zeros(1, ny);  
y_pid(1,:) = C_total*x0;

for k = 2:N
    e = ref(k-1,:) - y_pid(k-1,:);
    int_error = int_error + e*Ts;
    u_pid(k,:) = Kp*e' + Ki*int_error';
    u_pid(k,:) = max(min(u_pid(k,:),3),-3);  % Saturación
    x0 = A_opt*x0 + B_opt*u_pid(k,:)';
    y_pid(k,:) = (C_total*x0 + D_total*u_pid(k,:)')';
end

%% ------------------------------------------------------------------------
%% 6) EVALUACIÓN CUANTITATIVA
% -------------------------------------------------------------------------
fprintf('\n=== MPC robusto ===\n');
evaluate_mpc(y_mpc,ref);

fprintf('\n=== MPC simple ===\n');
evaluate_mpc(y_baseline,ref);

fprintf('\n=== PID MIMO respetable ===\n');
evaluate_mpc(y_pid,ref);

%% ------------------------------------------------------------------------
%% 7) FIGURAS
% -------------------------------------------------------------------------
figure('Color','w','Position',[100 100 1000 700]);

subplot(2,1,1)
plot(t,y_mpc,'LineWidth',1.5); hold on;
plot(t,y_baseline,'--','LineWidth',1.3);
plot(t,y_pid,':','LineWidth',1.3);
plot(t,ref,'k:','LineWidth',1.2);
title('Seguimiento: MPC robusto vs MPC simple vs PID MIMO');
legend('MPC y1','MPC y2','MPC simple y1','MPC simple y2','PID y1','PID y2','Ref','Location','best');
grid on;

subplot(2,1,2)
plot(t,vecnorm(y_mpc-ref,2,2),'LineWidth',1.5); hold on;
plot(t,vecnorm(y_baseline-ref,2,2),'--','LineWidth',1.3);
plot(t,vecnorm(y_pid-ref,2,2),':','LineWidth',1.3);
title('Error euclídeo de seguimiento');
legend('MPC robusto','MPC simple','PID MIMO');
xlabel('Tiempo [s]');
grid on;

sgtitle('Comparativa de controladores – Modelo reducido LSTM');

%% -------------------------------
%% 7b) FIGURAS: CONTROL INPUTS
%% -------------------------------
u1 = [u_mpc(:,1), u_baseline(:,1), u_pid(:,1)]; % Input 1 (Pump)
u2 = [u_mpc(:,2), u_baseline(:,2), u_pid(:,2)]; % Input 2 (Valve)
sat = [3 -3]; % Límites de saturación

figure('Color','w','Position',[100 100 1000 700]);

subplot(2,1,1); hold on; grid on;
plot(t, u1(:,1), 'b-', 'LineWidth', 2);
plot(t, u1(:,2), 'r--', 'LineWidth', 2);
plot(t, u1(:,3), 'k:', 'LineWidth', 2);
yline(sat(1), 'Color',[0.5 0.5 0.5],'LineStyle','-.','LineWidth',1.5);
yline(sat(2), 'Color',[0.5 0.5 0.5],'LineStyle','-.','LineWidth',1.5);
xlabel('Tiempo [s]'); ylabel('u_1(k)');
title('Control input u_1 (Pump)');
legend('MPC robusto','MPC simple','PID','Saturación');

subplot(2,1,2); hold on; grid on;
plot(t, u2(:,1), 'b-', 'LineWidth', 2);
plot(t, u2(:,2), 'r--', 'LineWidth', 2);
plot(t, u2(:,3), 'k:', 'LineWidth', 2);
yline(sat(1), 'Color',[0.5 0.5 0.5],'LineStyle','-.','LineWidth',1.5);
yline(sat(2), 'Color',[0.5 0.5 0.5],'LineStyle','-.','LineWidth',1.5);
xlabel('Tiempo [s]'); ylabel('u_2(k)');
title('Control input u_2 (Valve)');
legend('MPC robusto','MPC simple','PID','Saturación');

sgtitle('Comparativa de controladores – Inputs de control');

%% ------------------------------------------------------------------------
%% 7c) FIGURA FUNDAMENTAL: Referencia vs Salida (Tracking)
%% ------------------------------------------------------------------------
figure('Color','w','Position',[100 100 900 600]);

% --- Output y1 ---
subplot(2,1,1); hold on; grid on;
plot(t, ref(:,1), 'k--', 'LineWidth', 2);
plot(t, y_mpc(:,1), 'b-', 'LineWidth', 2);
plot(t, y_baseline(:,1), 'r-.', 'LineWidth', 1.8);
plot(t, y_pid(:,1), 'g:', 'LineWidth', 1.8);
ylabel('y_1');
title('Tracking performance for output y_1');
legend('Reference r_1','Robust MPC','Simple MPC','PID','Location','best');

% --- Output y2 ---
subplot(2,1,2); hold on; grid on;
plot(t, ref(:,2), 'k--', 'LineWidth', 2);
plot(t, y_mpc(:,2), 'b-', 'LineWidth', 2);
plot(t, y_baseline(:,2), 'r-.', 'LineWidth', 1.8);
plot(t, y_pid(:,2), 'g:', 'LineWidth', 1.8);
xlabel('Time [s]');
ylabel('y_2');
title('Tracking performance for output y_2');
legend('Reference r_2','Robust MPC','Simple MPC','PID','Location','best');

sgtitle('Reference tracking: r(t) vs y(t)');


%% ------------------------------------------------------------------------
%% 8) COMPUTATIONAL COMPLEXITY (Q1-ready, con desviación típica)
% -------------------------------------------------------------------------
clear e all_median_robust all_median_simple all_median_pid
rng(0);                 % Reproducibilidad
N_test = 50;            % Número de pasos de control
Nrep  = 5;              % Repeticiones por paso
Nexec = 100;             % Número de ejecuciones completas

% --- Preasignación de arrays para estadísticas ---
all_median_robust = zeros(Nexec,1);
all_median_simple = zeros(Nexec,1);
all_median_pid    = zeros(Nexec,1);

for e = 1:Nexec
    fprintf('--- Execution %d / %d ---\n', e, Nexec);

    % --- Preasignación de tiempos ---
    time_mpc_robust_all = zeros(N_test,Nrep);
    time_mpc_simple_all = zeros(N_test,Nrep);
    time_pid_all        = zeros(N_test,Nrep);

    % --- Inicializar estados ---
    x0_test = pinv(C_total)*ref(1,:)';
    x_plant_mpc    = x0_test;
    x_plant_simple = x0_test;
    x_plant_pid    = x0_test;

    x_mpc_robust = mpcstate(mpc_controller);
    x_mpc_simple = mpcstate(mpc_baseline);
    int_error = zeros(1, ny);  % PID integral

    % --- Warm-up del solver ---
    for k = 1:10
        mpcmove(mpc_controller, x_mpc_robust, ref(1,:)');
        mpcmove(mpc_baseline, x_mpc_simple, ref(1,:)');
    end

    % --- Medición de tiempos por paso ---
    for k = 2:N_test
        % Robust MPC
        for r = 1:Nrep
            tic;
            u = mpcmove(mpc_controller, x_mpc_robust, ref(k,:)');
            time_mpc_robust_all(k,r) = toc*1000;
        end
        x_plant_mpc = A_opt*x_plant_mpc + B_opt*u;

        % Simple MPC
        for r = 1:Nrep
            tic;
            u = mpcmove(mpc_baseline, x_mpc_simple, ref(k,:)');
            time_mpc_simple_all(k,r) = toc*1000;
        end
        x_plant_simple = A_opt*x_plant_simple + B_opt*u;

        % PID MIMO
        for r = 1:Nrep
            tic;
            e_pid = ref(k-1,:) - C_total*x_plant_pid;
            int_error = int_error + e_pid*Ts;
            u = Kp*e_pid' + Ki*int_error';
            u = max(min(u,3),-3);
            time_pid_all(k,r) = toc*1000;
        end
        x_plant_pid = A_opt*x_plant_pid + B_opt*u;
    end

    %% --- Estadísticos de tiempo por ejecución ---
    all_median_robust(e) = median(time_mpc_robust_all(2:end,:),'all');
    all_median_simple(e) = median(time_mpc_simple_all(2:end,:),'all');
    all_median_pid(e)    = median(time_pid_all(2:end,:),'all');
end

%% --- Mediana global y desviación típica ---
median_robust = mean(all_median_robust); 
std_robust    = std(all_median_robust);

median_simple = mean(all_median_simple);
std_simple    = std(all_median_simple);

median_pid    = mean(all_median_pid);
std_pid       = std(all_median_pid);

% Vector de barras y desviaciones
meds = [median_robust, median_simple, median_pid];
stds = [std_robust, std_simple, std_pid];

% --- Mostrar resultados en la consola ---
methods = {'Robust','Simple','PID'};
for i = 1:3
    fprintf('%s: Median = %.4f ms, Std = %.4f ms\n', methods{i}, meds(i), stds(i));
end


%% --- Preparar gráfico ---
x = [0.1, 10, 19];
figure; hold on;

% Tonos azul navy
colors = [0.0 0.2 0.4; 0.0 0.3 0.6; 0.0 0.4 0.8];
b = bar(x, meds, 'FaceColor','flat', 'BarWidth',0.5);
for i = 1:3
    b.CData(i,:) = colors(i,:);
end

xticks(x);
xticklabels({'Robust', 'Simple', 'PID'});
ylabel('Median [ms]');
title('Method Comparison');
set(gca, 'YScale', 'log');

% --- Bigotes de desviación típica bastante a la derecha de la barra ---
line_width = 1;
dx = 3;  % desplazamiento horizontal grande a la derecha
for i = 1:3
    sigma = stds(i);             % desviación típica
    upper = meds(i) + sigma;     % extremo superior del bigote
    lower = meds(i) - sigma;     % extremo inferior del bigote

    % Bigote vertical a la derecha de la barra
    plot([x(i)+dx, x(i)+dx], [lower, upper], 'Color', [0.8 0 0], 'LineWidth', line_width);

    % Líneas horizontales en extremos del bigote
    plot([x(i)+dx-0.3, x(i)+dx+0.3], [upper, upper], 'Color', [0.8 0 0], 'LineWidth', line_width);
    plot([x(i)+dx-0.3, x(i)+dx+0.3], [lower, lower], 'Color', [0.8 0 0], 'LineWidth', line_width);

    % Mostrar mediana sobre la barra
    text(x(i), meds(i)*1.15, sprintf('t̄: %.4f', meds(i)), ...
         'HorizontalAlignment','center', 'FontSize',11, 'FontWeight','bold', 'Color',[0 0.1 0.3]);

    % Mostrar desviación típica a la derecha del bigote superior
    text(x(i)+dx+0.2, upper, sprintf('σ: %.4f', sigma), ...
     'FontSize', 10, 'VerticalAlignment', 'bottom', 'Color', [1 0 0]);

end

hold off;






%% ========================================================================
%% FUNCIÓN DE EVALUACIÓN (Q1)
% ========================================================================
function evaluate_mpc(y,r)

    e = y-r;
    rmse = sqrt(mean(e.^2));
    mae  = mean(abs(e));
    iae  = sum(abs(e));
    ise  = sum(e.^2);

    for i=1:size(y,2)
        fprintf('Salida %d | RMSE=%.4f | MAE=%.4f | IAE=%.2f | ISE=%.2f | ef=%.4f\n',...
            i,rmse(i),mae(i),iae(i),ise(i),abs(e(end,i)));
    end
    fprintf('RMSE global = %.4f\n',sqrt(mean(e(:).^2)));
end
