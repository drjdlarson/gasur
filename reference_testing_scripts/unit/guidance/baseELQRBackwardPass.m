function [x_hat, u_hat, feedback, feedforward, cost_go_mat, ...
    cost_go_vec] =  baseELQRBackwardPass()


Q_nom = 10^-3 * eye(2);
R_nom = 0.1;
x_hat = [0; 1];
x_start = x_hat;
u_nom = 1;
feedback = ones(1, 2, 3);
feedforward = ones(1, 1, 3);
cost_come_mat = ones(2, 2, 3);
cost_come_vec = ones(2, 1, 3);
cost_go_mat = ones(2, 2, 3);
cost_go_vec = ones(2, 1, 3);

dyn_fnc1 = @(x, u) 2*x(1) / (1 - x(1)) + u(1);
dyn_fnc2 = @(x, u) (1 - 2*x(2)) / 5 - u(1)^2;

inv_fnc1 = @(x, u) (x(1) - u(1)) / (x(1) + 2 + u(1));
inv_fnc2 = @(x, u) 0.5 * (1 - 5*x(2) + 5*u(1)^2);

%% function
max_time_steps = size(cost_come_mat, 3);
for kk = max_time_steps-1:-1:1
    u_hat = feedback(:, :, kk) * x_hat + feedforward(:, :, kk);
    x_hat_prime = [0; 0];
    x_hat_prime(1) = inv_fnc1(x_hat, u_hat);
    x_hat_prime(2) = inv_fnc2(x_hat, u_hat);

    state_mat_bar = get_state_jacobian(x_hat_prime, u_hat, ...
                                       {dyn_fnc1, dyn_fnc2});
    input_mat_bar = get_input_jacobian(x_hat_prime, u_hat, ...
                                       {dyn_fnc1, dyn_fnc2});
    c_bar_vec = x_hat - state_mat_bar * x_hat_prime ...
        - input_mat_bar * u_hat;

    [P, Q, R, q, r] = quadratize_cost(x_hat_prime, u_hat, x_start, u_nom, kk, ...
        Q_nom, R_nom);

    [cost_go_mat(:, :, kk), cost_go_vec(:, :, kk), ...
        feedback(:, :, kk), feedforward(:, :, kk)] ...
        = cost_to_go(cost_go_mat(:, :, kk+1), cost_go_vec(:, :, kk+1), ...
        P, Q, R, q, r, state_mat_bar, input_mat_bar, c_bar_vec);

    x_hat = -inv(cost_go_mat(:, :, kk) + cost_come_mat(:, :, kk)) ...
        * (cost_go_vec(:, :, kk) + cost_come_vec(:, :, kk));
end

end

function A = get_state_jacobian(x, u, fncs)
step_size = 1e-7;
inv_step2 = 1 / (2 * step_size);
n_states = numel(x);
A = zeros(n_states);
for row = 1:n_states
    for col = 1:n_states
        x_r = x;
        x_l = x;
        x_r(col) = x_r(col) + step_size;
        x_l(col) = x_l(col) - step_size;
        A(row, col) = inv_step2 * (fncs{row}(x_r, u) ...
                                   - fncs{row}(x_l, u));
    end
end
end

function B = get_input_jacobian(x, u, fncs)
step_size = 1e-7;
inv_step2 = 1 / (2 * step_size);
n_states = numel(x);
n_inputs = numel(u);
B = zeros(n_states, n_inputs);
for row = 1:n_states
    for col = 1:n_inputs
        u_r = u;
        u_l = u;
        u_r(col) = u_r(col) + step_size;
        u_l(col) = u_l(col) - step_size;
        B(row, col) = inv_step2 * (fncs{row}(x, u_r) ...
                                   - fncs{row}(x, u_l));
    end
end
end

function [P, Q, R, q, r] = quadratize_cost(x, u, x_start, u_nom, kk, Q, R)
if kk == 1
    Q = Q;
    q = -Q * x_start;
else
    Q = Q;
    q = zeros(2, 1);
end

R = R;
r = -R * u_nom;
P = zeros(1, 2);


end

function [cost_go_mat_out, cost_go_vec_out, feedback, ...
    feedforward] = cost_to_go(cost_go_mat, cost_go_vec, ...
    P, Q, R, q, r, state_mat_bar, input_mat_bar, c_bar_vec)

c_mat = input_mat_bar' * cost_go_mat * state_mat_bar + P;
d_mat = state_mat_bar' * cost_go_mat * state_mat_bar + Q;
e_mat = input_mat_bar' * cost_go_mat * input_mat_bar + R;
tmp = (cost_go_vec + cost_go_mat * c_bar_vec);
d_vec = state_mat_bar' * tmp + q;
e_vec = input_mat_bar' * tmp + r;

e_inv = inv(e_mat);
feedback = -e_inv * c_mat;
feedforward = -e_inv * e_vec;

cost_go_mat_out = d_mat + c_mat' * feedback;
cost_go_vec_out = d_vec + c_mat' * feedforward;

end
