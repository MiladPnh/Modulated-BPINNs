# train_darcy_pinn.py
import tensorflow as tf
import numpy as np
from scipy.stats import qmc
import time
import os
import argparse
from pinn_architecture import PirateNet # Import your model

# TensorFlow configurations (optional, can be set based on environment)
# tf.config.experimental.enable_mlir_graph_optimization()
# tf.config.optimizer.set_experimental_options({'layout_optimizer': True})
# tf.config.optimizer.set_jit(True)

# --- Physics and Utility Functions ---
@tf.function
def grf(x, y, k_params_grf): # Renamed k to k_params_grf for clarity
    dx2 = tf.square(x - k_params_grf[:,0:1])
    dy2 = tf.square(y - k_params_grf[:,1:2])
    hw = tf.constant(0.15, dtype=tf.float32)
    exp_neg = tf.exp(tf.constant(-1.5, dtype=tf.float32))
    exp_pos = tf.exp(tf.constant(1.5, dtype=tf.float32))
    exponent = - (dx2/(2*hw*hw) + dy2/(2*hw*hw))
    return exp_neg + (exp_pos - exp_neg) * tf.exp(exponent)

@tf.function
def grf_i(x, y, k_params_grf_i): # Renamed k
    regional_baseline_head = tf.constant(0.5, dtype=tf.float32)
    head_variation_peak = tf.constant(-0.4, dtype=tf.float32)
    variation_width_factor = tf.constant(0.15, dtype=tf.float32)
    center_x = k_params_grf_i[:, 0:1]
    center_y = k_params_grf_i[:, 1:2]
    dx2 = tf.square(x - center_x)
    dy2 = tf.square(y - center_y)
    exponent = - (dx2 / (2 * tf.square(variation_width_factor)) + \
                  dy2 / (2 * tf.square(variation_width_factor)))
    gaussian_influence = tf.exp(exponent)
    initial_head = regional_baseline_head + head_variation_peak * gaussian_influence
    return initial_head

@tf.function
def v_model(current_h_model, x, y, k_params_v, t): # Renamed k
    with tf.GradientTape() as tape:
        tape.watch([x, y])
        # Input to h_model should be [x, y, k_params_v, t]
        h_input = tf.concat([x, y, k_params_v, t], axis=1)
        h = current_h_model(h_input)
        
        # Conductivity field a(x,y; k)
        conductivity = grf(x, y, k_params_v) # Use k_params_v for conductivity too

    dh_dx, dh_dy = tape.gradient(h, [x,y])
    
    dh_dx = tf.zeros_like(x) if dh_dx is None else dh_dx
    dh_dy = tf.zeros_like(y) if dh_dy is None else dh_dy
    
    return -conductivity * dh_dx, -conductivity * dh_dy

@tf.function
def h_f_model(current_h_model, x, y, k_params_hf, t, T_phys_scale_factor, source_term_beta): # Renamed k
    T_s = tf.constant(T_phys_scale_factor, dtype=tf.float32)
    beta_val = tf.constant(source_term_beta, dtype=tf.float32) # Renamed beta to beta_val

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, y])
        a_field = grf(x, y, k_params_hf) # Conductivity
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y, t])
            h_input = tf.concat([x, y, k_params_hf, t], axis=1)
            h = current_h_model(h_input)

        h_x = tape1.gradient(h, x)
        h_y = tape1.gradient(h, y)
        h_t = tape1.gradient(h, t)
        del tape1

        h_x = tf.zeros_like(x) if h_x is None else h_x
        h_y = tf.zeros_like(y) if h_y is None else h_y
        h_t = tf.zeros_like(t) if h_t is None else h_t
        
        flux_x_component = a_field * h_x
        flux_y_component = a_field * h_y

    div_a_grad_h_x = tape2.gradient(flux_x_component, x)
    div_a_grad_h_y = tape2.gradient(flux_y_component, y)
    del tape2
    
    div_a_grad_h_x = tf.zeros_like(x) if div_a_grad_h_x is None else div_a_grad_h_x
    div_a_grad_h_y = tf.zeros_like(y) if div_a_grad_h_y is None else div_a_grad_h_y

    divergence_term = div_a_grad_h_x + div_a_grad_h_y
    h_f_residual = h_t - T_s*(divergence_term + beta_val) # Renamed h_f to h_f_residual
    return h_f_residual

# Global variable for gradient balancing (initialized in main training function)
boundary_ratio_moving_avg = None

@tf.function
def calculate_loss_components(current_h_model,
                              x_f, y_f, k_f, t_f, T_phys_scale_factor, source_term_beta,
                              x_i, y_i, k_i, t_i,
                              x_db1, y_db1, k_db1, t_db1, # Bottom boundary part 1 (influx)
                              x_db2, y_db2, k_db2, t_db2, # Bottom boundary part 2 (no-flux)
                              x_ub, y_ub, k_ub, t_ub,    # Upper boundary (no-flux)
                              x_lb, y_lb, k_lb, t_lb,    # Left boundary (no-flux)
                              x_rb1, y_rb1, k_rb1, t_rb1, # Right boundary part 1 (h=0)
                              x_rb2, y_rb2, k_rb2, t_rb2, # Right boundary part 2 (no-flux)
                              weights_f, initial_condition_type='zero'):

    # Initial Condition
    if initial_condition_type == 'zero':
        h_i_target = tf.constant(0.0, dtype=tf.float32)
    elif initial_condition_type == 'grf':
        h_i_target = grf_i(x_i, y_i, k_i)
    else:
        h_i_target = tf.constant(0.0, dtype=tf.float32)
    h_i_pred = current_h_model(tf.concat([x_i, y_i, k_i, t_i], axis=1))
    loss_h_i = h_i_pred - h_i_target

    # PDE Residual
    loss_h_f = h_f_model(current_h_model, x_f, y_f, k_f, t_f, T_phys_scale_factor, source_term_beta)
    
    # Boundary Conditions
    # Left boundary (x=0): No-flux (Vx=0)
    vx_lb, _ = v_model(current_h_model, x_lb, y_lb, k_lb, t_lb)
    loss_h_lb = vx_lb - 0.0 
    
    # Right boundary part 1 (x=1, 0.8 <= y <= 1): Dirichlet h=0
    h_rb1_pred = current_h_model(tf.concat([x_rb1, y_rb1, k_rb1, t_rb1], axis=1))
    loss_h_rb1 = h_rb1_pred - 0.0
    
    # Right boundary part 2 (x=1, 0 <= y < 0.8): No-flux (Vx=0)
    vx_rb2, _ = v_model(current_h_model, x_rb2, y_rb2, k_rb2, t_rb2)
    loss_h_rb2 = vx_rb2 - 0.0
    
    # Upper boundary (y=1): No-flux (Vy=0)
    _, vy_ub = v_model(current_h_model, x_ub, y_ub, k_ub, t_ub)
    loss_h_ub = vy_ub - 0.0
    
    # Bottom boundary part 1 (y=0, 0 <= x <= 0.2): Parabolic Influx (Vy_target = -target_influx)
    # v_model returns positive Vy for upward flow. Influx is downward (negative Vy).
    # Target flux is positive INTO domain. So target_vy = -parabolic_flux_value
    _, vy_db1_pred = v_model(current_h_model, x_db1, y_db1, k_db1, t_db1)
    target_influx_val_db1 = 4.0 * (x_db1 / 0.2) * (1.0 - (x_db1 / 0.2)) # Positive for influx value
    loss_h_db1 = (vy_db1_pred) - (-target_influx_val_db1) # vy_pred should match -target_influx

    # Bottom boundary part 2 (y=0, 0.2 < x <= 1): No-flux (Vy=0)
    _, vy_db2_pred = v_model(current_h_model, x_db2, y_db2, k_db2, t_db2)
    loss_h_db2 = vy_db2_pred - 0.0
    
    # MSE for each component
    mse_i_h = tf.reduce_mean(tf.square(loss_h_i))
    mse_f_h = tf.reduce_mean(weights_f * tf.square(loss_h_f)) # Causal weights applied here
    mse_lb_h = tf.reduce_mean(tf.square(loss_h_lb))
    mse_rb1_h = tf.reduce_mean(tf.square(loss_h_rb1))
    mse_rb2_h = tf.reduce_mean(tf.square(loss_h_rb2))
    mse_db1_h = tf.reduce_mean(tf.square(loss_h_db1))
    mse_db2_h = tf.reduce_mean(tf.square(loss_h_db2))
    mse_ub_h = tf.reduce_mean(tf.square(loss_h_ub))
    
    mse_b_h = (mse_lb_h + mse_rb1_h + mse_rb2_h + mse_db1_h + mse_db2_h + mse_ub_h)
    mse_bi_h = mse_b_h + mse_i_h # Combined Boundary and Initial loss
    
    # For monitoring flux balance (rough estimate)
    # Actual influx at db1 segment (length 0.2)
    monitor_influx_db1 = tf.reduce_mean(-vy_db1_pred) * 0.2 
    # Actual outflux at rb1 segment (length 0.2 where h=0)
    vx_rb1_monitor, _ = v_model(current_h_model, x_rb1, y_rb1, k_rb1, t_rb1)
    monitor_outflux_rb1 = tf.reduce_mean(vx_rb1_monitor) * 0.2
    
    # Total loss for backpropagation
    total_loss = mse_bi_h + mse_f_h
    
    return (total_loss, mse_i_h, mse_b_h, mse_bi_h, mse_f_h,
            monitor_influx_db1, monitor_outflux_rb1)


@tf.function
def calculate_gradients_and_apply(current_h_model, adam_optimizer,
                                  optimizer_params, # Dict for T_phys_scale_factor, etc.
                                  training_data_dict, # Contains all x_f, y_f, k_f, t_f etc.
                                  weights_f,
                                  gradient_balancing_alpha,
                                  initial_condition_type):
    global boundary_ratio_moving_avg # Use the global tf.Variable

    with tf.GradientTape(persistent=True) as tape:
        loss_value, mse_i_h, mse_b_h, mse_bi_h, mse_f_h, influx_db, outflux_rb = \
            calculate_loss_components(
                current_h_model,
                training_data_dict['x_f'], training_data_dict['y_f'], training_data_dict['k_f'], training_data_dict['t_f'],
                optimizer_params['T_phys_scale_factor'],
                optimizer_params['source_term_beta'],
                training_data_dict['x_i'], training_data_dict['y_i'], training_data_dict['k_i'], training_data_dict['t_i'],
                training_data_dict['x_db1'], training_data_dict['y_db1'], training_data_dict['k_db1'], training_data_dict['t_db1'],
                training_data_dict['x_db2'], training_data_dict['y_db2'], training_data_dict['k_db2'], training_data_dict['t_db2'],
                training_data_dict['x_ub'], training_data_dict['y_ub'], training_data_dict['k_ub'], training_data_dict['t_ub'],
                training_data_dict['x_lb'], training_data_dict['y_lb'], training_data_dict['k_lb'], training_data_dict['t_lb'],
                training_data_dict['x_rb1'], training_data_dict['y_rb1'], training_data_dict['k_rb1'], training_data_dict['t_rb1'],
                training_data_dict['x_rb2'], training_data_dict['y_rb2'], training_data_dict['k_rb2'], training_data_dict['t_rb2'],
                weights_f,
                initial_condition_type
            )

    model_vars = current_h_model.trainable_variables
    grads_domain = tape.gradient(mse_f_h, model_vars)
    grads_boundary = tape.gradient(mse_bi_h, model_vars) # Gradient of (BC+IC loss)
    del tape

    clean_grads_domain = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads_domain, model_vars)]
    clean_grads_boundary = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads_boundary, model_vars)]

    G_n_domain = tf.linalg.global_norm(clean_grads_domain)
    G_n_boundary = tf.linalg.global_norm(clean_grads_boundary)
    
    ratio = G_n_boundary / (G_n_domain + 1e-10)
    
    alpha_gb_tf = tf.constant(gradient_balancing_alpha, dtype=tf.float32)
    current_avg = boundary_ratio_moving_avg
    new_avg = current_avg * (1.0 - alpha_gb_tf) + alpha_gb_tf * ratio
    
    new_avg_clipped = tf.clip_by_value(new_avg, 0.01, 100.)
    boundary_ratio_moving_avg.assign(new_avg_clipped)
    
    domain_scale = tf.where(new_avg_clipped >= 1.0, new_avg_clipped, 1.0)
    boundary_scale = tf.where(new_avg_clipped < 1.0, 1.0 / new_avg_clipped, 1.0)
    
    combined_grads = [domain_scale * g1 + boundary_scale * g2 
                      for g1, g2 in zip(clean_grads_domain, clean_grads_boundary)]
    
    adam_optimizer.apply_gradients(zip(combined_grads, model_vars))
    
    G_n_combined = tf.linalg.global_norm(combined_grads)

    return (loss_value, mse_f_h, mse_b_h, mse_i_h,
            G_n_domain, G_n_boundary, G_n_combined,
            new_avg_clipped, influx_db, outflux_rb)


class WarmUpExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_rate, decay_steps, name=None):
        super(WarmUpExponentialDecay, self).__init__() # Added super call
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        exp_lr = self.initial_lr * tf.pow(
            self.decay_rate, ((step - self.warmup_steps) / self.decay_steps)
        )
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: exp_lr)

    def get_config(self):
        return {
            "initial_lr": float(self.initial_lr.numpy()), # Convert EagerTensor to float
            "warmup_steps": int(self.warmup_steps.numpy()),
            "decay_rate": float(self.decay_rate.numpy()),
            "decay_steps": int(self.decay_steps.numpy()),
            "name": self.name
        }

# --- Main Training Function ---
def run_training(args):
    global boundary_ratio_moving_avg # Declare we are using the global one
    boundary_ratio_moving_avg = tf.Variable(1.0, trainable=False, dtype=tf.float32)


    print("Starting training with configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 70)

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
        # Potentially set Python's random seed too if used for other stochasticity
        # import random
        # random.seed(args.seed)

    # Calculate dependent parameters
    actual_input_dim = 2 + args.N_g + 1  # x, y, k_params (N_g), t
    epochs_per_interval = args.total_epochs // args.N_chunks
    if args.total_epochs < args.N_chunks:
        print("Warning: total_epochs < N_chunks. Adjusting N_chunks to total_epochs.")
        args.N_chunks = args.total_epochs
        epochs_per_interval = 1


    # Initialize Model
    h_model_instance = PirateNet(
        input_dim=actual_input_dim,
        output_dim=args.output_dim,
        m=args.piratenet_m,
        s_init_val=args.piratenet_s,
        L=args.piratenet_L,
        activation_fn_str=args.piratenet_activation,
        RWFactorized=args.piratenet_rwf
    )
    h_model_instance.summary(line_length=120)

    if args.load_weights_path:
        if os.path.exists(args.load_weights_path):
            print(f"Loading model weights from: {args.load_weights_path}")
            h_model_instance.load_weights(args.load_weights_path)
        else:
            print(f"Warning: Specified load_weights_path not found: {args.load_weights_path}. Starting from scratch.")


    # Optimizer and LR Schedule
    learning_rate_schedule = WarmUpExponentialDecay(
        initial_lr=args.lr_initial,
        warmup_steps=args.lr_warmup_steps,
        decay_rate=args.lr_decay_rate,
        decay_steps=args.lr_decay_steps,
        name="WarmUpExponentialDecayScheduler"
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    optimizer_params_dict = {
        'T_phys_scale_factor': args.T_phys_scale_factor,
        'source_term_beta': args.source_term_beta
    }

    # Sobol Engines
    s_engine_spatial = qmc.Sobol(d=3, scramble=True, seed=args.seed) # x, y, t_unscaled
    s_engine_bc = qmc.Sobol(d=3, scramble=True, seed=args.seed + 1 if args.seed is not None else None) # x_b, y_b, t_unscaled
    s_engine_latent = qmc.Sobol(d=args.N_g, scramble=True, seed=args.seed + 2 if args.seed is not None else None)


    global_epoch_counter = args.start_epoch # Use for continuing training and logging
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output directory exists

    print(f"Training will run for {args.total_epochs} total epochs, split into {args.N_chunks} causal intervals.")
    print(f"Each interval will have {epochs_per_interval} epochs.")

    for i_interval in range(1, args.N_chunks + 1):
        current_T_for_interval = args.T_final * i_interval / args.N_chunks
        interval_start_time_py = time.time()
        print(f"\n--- Starting Causal Interval {i_interval}/{args.N_chunks} (Time up to {current_T_for_interval:.4f}) ---")

        for epoch_in_interval in range(epochs_per_interval):
            if global_epoch_counter >= args.total_epochs:
                print("Reached total_epochs limit. Stopping training.")
                break

            # --- Data Generation for current batch ---
            latent_vectors_batch = s_engine_latent.random(args.N_latent_batch).astype(np.float32)
            # Select a subset of these latent vectors to combine with spatial points
            chosen_latent_indices = np.random.choice(latent_vectors_batch.shape[0], args.N_latent_samples_per_batch, replace=False)
            sobol_k_samples = latent_vectors_batch[chosen_latent_indices]

            # Generate spatial & unscaled time points
            sobol_xy_t_f = s_engine_spatial.random(args.N_spatial_f).astype(np.float32)
            sobol_xy_t_b = s_engine_bc.random(args.N_spatial_bi).astype(np.float32)

            # Tile/Repeat to combine with latent k samples
            k_f_tiled = np.repeat(sobol_k_samples, args.N_spatial_f, axis=0)
            xy_t_f_tiled = np.tile(sobol_xy_t_f, (args.N_latent_samples_per_batch, 1))
            XF_full = np.hstack((xy_t_f_tiled[:, :2], k_f_tiled, xy_t_f_tiled[:, 2:3])) # x,y,k,t_unscaled

            k_b_tiled = np.repeat(sobol_k_samples, args.N_spatial_bi, axis=0)
            xy_t_b_tiled = np.tile(sobol_xy_t_b, (args.N_latent_samples_per_batch, 1))
            XB_full = np.hstack((xy_t_b_tiled[:, :2], k_b_tiled, xy_t_b_tiled[:, 2:3])) # x,y,k,t_unscaled
            
            # --- Create data dictionary and apply transformations ---
            # Collocation points (time scaled within interval)
            # Add small epsilon to prevent t_f from being exactly 0 if current_T is small and random samples are 0
            t_f_scaled = (-0.05 + (current_T_for_interval + 0.05) * XF_full[:, -1:]).astype(np.float32) 
            training_data = {'x_f': XF_full[:, 0:1], 'y_f': XF_full[:, 1:2], 
                             'k_f': XF_full[:, 2:2+args.N_g], 't_f': t_f_scaled}

            # Initial condition points (t=0)
            training_data.update({'x_i': XB_full[:, 0:1], 'y_i': XB_full[:, 1:2],
                                  'k_i': XB_full[:, 2:2+args.N_g], 't_i': np.zeros_like(XB_full[:, -1:])})

            # Boundary condition points (time scaled like t_f)
            t_b_scaled = (-0.05 + (current_T_for_interval + 0.05) * XB_full[:, -1:]).astype(np.float32)

            # Define BC points by modifying coordinates from XB_full and using t_b_scaled
            # x_lb, y_lb, k_lb, t_lb (Left Boundary: x=0, no-flux)
            training_data['x_lb'] = np.zeros_like(XB_full[:, 0:1])
            training_data['y_lb'] = XB_full[:, 1:2]
            training_data['k_lb'] = XB_full[:, 2:2+args.N_g]
            training_data['t_lb'] = t_b_scaled

            # x_rb1, y_rb1, k_rb1, t_rb1 (Right Boundary part 1: x=1, 0.8 <= y <= 1, h=0)
            training_data['x_rb1'] = np.ones_like(XB_full[:, 0:1])
            training_data['y_rb1'] = 0.8 + 0.2 * XB_full[:, 1:2] # Scale y to [0.8, 1.0]
            training_data['k_rb1'] = XB_full[:, 2:2+args.N_g]
            training_data['t_rb1'] = t_b_scaled
            
            # x_rb2, y_rb2, k_rb2, t_rb2 (Right Boundary part 2: x=1, 0 <= y < 0.8, no-flux)
            training_data['x_rb2'] = np.ones_like(XB_full[:, 0:1])
            training_data['y_rb2'] = 0.8 * XB_full[:, 1:2] # Scale y to [0, 0.8)
            training_data['k_rb2'] = XB_full[:, 2:2+args.N_g]
            training_data['t_rb2'] = t_b_scaled

            # x_db1, y_db1, k_db1, t_db1 (Bottom Boundary part 1: y=0, 0 <= x <= 0.2, influx)
            training_data['x_db1'] = 0.2 * XB_full[:, 0:1] # Scale x to [0, 0.2]
            training_data['y_db1'] = np.zeros_like(XB_full[:, 1:2])
            training_data['k_db1'] = XB_full[:, 2:2+args.N_g]
            training_data['t_db1'] = t_b_scaled
            
            # x_db2, y_db2, k_db2, t_db2 (Bottom Boundary part 2: y=0, 0.2 < x <= 1, no-flux)
            training_data['x_db2'] = 0.2 + 0.8 * XB_full[:, 0:1] # Scale x to (0.2, 1.0]
            training_data['y_db2'] = np.zeros_like(XB_full[:, 1:2])
            training_data['k_db2'] = XB_full[:, 2:2+args.N_g]
            training_data['t_db2'] = t_b_scaled
            
            # x_ub, y_ub, k_ub, t_ub (Upper Boundary: y=1, no-flux)
            training_data['x_ub'] = XB_full[:, 0:1]
            training_data['y_ub'] = np.ones_like(XB_full[:, 1:2])
            training_data['k_ub'] = XB_full[:, 2:2+args.N_g]
            training_data['t_ub'] = t_b_scaled

            # Convert all numpy arrays in dictionary to TensorFlow constants
            for key in training_data:
                training_data[key] = tf.constant(training_data[key], dtype=tf.float32)
            
            # --- Calculate Causal Weights ---
            safe_current_T = current_T_for_interval + 1e-8 # Denominator for weights
            # Using i_interval ensures that lambda effectively decreases as training progresses through time
            weights_f = tf.exp(-args.Causal_tolerance / i_interval * training_data['t_f'] / safe_current_T)
            # Optional: Normalize weights_f = weights_f / (tf.reduce_mean(weights_f) + 1e-8)

            # --- Perform one optimization step ---
            loss_val, mse_f, mse_b, mse_i, G_dom, G_bound, G_comb, \
            boundary_ratio_val, influx, outflux = calculate_gradients_and_apply(
                h_model_instance, adam_optimizer, optimizer_params_dict,
                training_data, weights_f,
                args.gradient_balancing_alpha,
                args.initial_condition_type
            )

            if global_epoch_counter % args.log_frequency == 0:
                current_lr = adam_optimizer.lr(global_epoch_counter).numpy() # Get current LR
                elapsed_interval_py = time.time() - interval_start_time_py
                tf.print(f"Global Ep: {global_epoch_counter} (Int {i_interval}, Ep {epoch_in_interval+1}/{epochs_per_interval}), "
                         f"LR: {current_lr:.2e}, Time/Int: {elapsed_interval_py:.1f}s")
                tf.print(f"  Losses - Tot: {loss_val:.3e}, PDE: {mse_f:.3e}, BC: {mse_b:.3e}, IC: {mse_i:.3e}")
                tf.print(f"  Fluxes - In: {influx:.3e}, Out: {outflux:.3e} (Monitored)")
                tf.print(f"  GradNorms - Dom: {G_dom:.2e}, Bnd: {G_bound:.2e}, Comb: {G_comb:.2e}")
                tf.print(f"  Balancing Ratio (Avg): {boundary_ratio_val:.3f}")
                # for blk_idx, block in enumerate(h_model_instance.blocks):
                #     tf.print(f"  Block {blk_idx+1} alpha: {block.alpha.numpy():.4f}", end=' ')
                # tf.print("\n" + "-" * 80)
            
            global_epoch_counter += 1

            if global_epoch_counter % args.save_frequency == 0 and global_epoch_counter > 0:
                model_filename = f"{args.model_save_name_prefix}_ep{global_epoch_counter}.weights.h5" # Keras preferred format
                model_save_path = os.path.join(args.output_dir, model_filename)
                print(f"\nSaving model weights at global epoch {global_epoch_counter} to {model_save_path}\n")
                h_model_instance.save_weights(model_save_path)
        
        # End of epochs in interval
        interval_duration_py = time.time() - interval_start_time_py
        print(f"--- Finished Causal Interval {i_interval}/{args.N_chunks} (Time up to {current_T_for_interval:.4f}) --- Duration: {interval_duration_py:.2f}s ---")
        # Optionally save model at end of each interval
        if args.save_at_interval_end:
            model_filename = f"{args.model_save_name_prefix}_interval{i_interval}_ep{global_epoch_counter}.weights.h5"
            model_save_path = os.path.join(args.output_dir, model_filename)
            print(f"\nSaving model weights at end of interval {i_interval} to {model_save_path}\n")
            h_model_instance.save_weights(model_save_path)


    print(f"\nTotal training completed after {global_epoch_counter} global epochs.")
    final_model_filename = f"{args.model_save_name_prefix}_final_ep{global_epoch_counter}.weights.h5"
    final_model_save_path = os.path.join(args.output_dir, final_model_filename)
    print(f"Saving final model weights to {final_model_save_path}")
    h_model_instance.save_weights(final_model_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a PirateNet PINN for Darcy Flow with Causal Training.")

    # Model Architecture arguments
    parser.add_argument('--output_dim', type=int, default=1, help="Output dimension of the PINN (typically 1 for head h).")
    parser.add_argument('--piratenet_m', type=int, default=128, help="Number of random Fourier features (half of embedding dim).")
    parser.add_argument('--piratenet_s', type=float, default=5.0, help="Stddev for sampling B matrix in PirateNet.")
    parser.add_argument('--piratenet_L', type=int, default=3, help="Number of residual blocks in PirateNet.")
    parser.add_argument('--piratenet_activation', type=str, default='tanh', help="Activation function for PirateNet (e.g., 'tanh', 'swish').")
    parser.add_argument('--piratenet_rwf', type=lambda x: (str(x).lower() == 'true'), default=True, help="Use RandomWeightFactorized layers (True/False).")

    # Training Loop arguments
    parser.add_argument('--total_epochs', type=int, default=640000, help="Total number of training epochs.")
    parser.add_argument('--T_final', type=float, default=1.0, help="Final time for the simulation training range.")
    parser.add_argument('--N_chunks', type=int, default=16, help="Number of causal time intervals (chunks).")
    parser.add_argument('--Causal_tolerance', type=float, default=1.0, help="Lambda parameter for causal weighting.")
    parser.add_argument('--initial_condition_type', type=str, default='zero', choices=['zero', 'grf'], help="Type of initial condition for head.")

    # Optimizer & Loss arguments
    parser.add_argument('--lr_initial', type=float, default=1e-3, help="Initial learning rate after warmup.")
    parser.add_argument('--lr_warmup_steps', type=int, default=5000, help="Number of warmup steps for learning rate.")
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help="Exponential decay rate for learning rate.")
    parser.add_argument('--lr_decay_steps', type=int, default=12500, help="Steps over which to apply one decay of LR.") # total_epochs / (N_chunks*2) ~ 2500 per interval
    parser.add_argument('--T_phys_scale_factor', type=float, default=5.0, help="Physical time scaling factor in PDE residual.")
    parser.add_argument('--source_term_beta', type=float, default=0.0, help="Source term beta in PDE residual.")
    parser.add_argument('--gradient_balancing_alpha', type=float, default=0.1, help="Alpha for moving average in gradient balancing.")

    # Data Generation arguments
    parser.add_argument('--N_g', type=int, default=2, help="Dimension of the latent vector k (conductivity parameters).")
    parser.add_argument('--N_spatial_f', type=int, default=2**12, help="Number of spatial collocation points per latent sample batch.")
    parser.add_argument('--N_spatial_bi', type=int, default=2**10, help="Number of spatial boundary/initial points per latent sample batch.")
    parser.add_argument('--N_latent_batch', type=int, default=2**8, help="Total number of k-vectors sampled by Sobol per epoch.")
    parser.add_argument('--N_latent_samples_per_batch', type=int, default=2**3, help="Number of k-vectors actually used to tile with spatial points for a training batch.")

    # IO & Logging arguments
    parser.add_argument('--output_dir', type=str, default='training_output', help="Directory to save models and logs.")
    parser.add_argument('--model_save_name_prefix', type=str, default='darcy_causal_piratenet', help="Prefix for saved model files.")
    parser.add_argument('--log_frequency', type=int, default=2000, help="Frequency (in epochs) to log training progress.")
    parser.add_argument('--save_frequency', type=int, default=50000, help="Frequency (in epochs) to save model weights.")
    parser.add_argument('--save_at_interval_end', type=lambda x: (str(x).lower() == 'true'), default=False, help="Save model at the end of each causal interval.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--start_epoch', type=int, default=0, help="Global epoch to start/resume training from.")
    parser.add_argument('--load_weights_path', type=str, default=None, help="Path to model weights (.h5) to load for resuming training.")

    args = parser.parse_args()
    
    # Calculate input_dim based on N_g after parsing
    args.input_dim = 2 + args.N_g + 1 # x, y, k_params (N_g), t

    run_training(args)