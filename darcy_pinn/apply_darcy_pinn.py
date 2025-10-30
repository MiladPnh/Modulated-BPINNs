# DarcyPINN_Project/darcy_pinn/apply_darcy_pinn.py
import tensorflow as tf
import numpy as np
import os
import argparse
from .pinn_architecture import PirateNet # Assuming it's in the same package

# --- Physics Functions (copied/adapted from train_darcy_pinn.py for standalone use) ---
# It's good practice to have these in a shared utility module if they grow complex.
@tf.function
def grf(x, y, k_params_grf):
    dx2 = tf.square(x - k_params_grf[:,0:1])
    dy2 = tf.square(y - k_params_grf[:,1:2])
    hw = tf.constant(0.15, dtype=tf.float32)
    exp_neg = tf.exp(tf.constant(-1.5, dtype=tf.float32))
    exp_pos = tf.exp(tf.constant(1.5, dtype=tf.float32))
    exponent = - (dx2/(2*hw*hw) + dy2/(2*hw*hw))
    return exp_neg + (exp_pos - exp_neg) * tf.exp(exponent)

@tf.function
def v_model_from_h(current_h_model, x, y, k_params_v, t):
    with tf.GradientTape() as tape:
        tape.watch([x, y])
        h_input = tf.concat([x, y, k_params_v, t], axis=1)
        h_pred = current_h_model(h_input)
        conductivity = grf(x, y, k_params_v)

    dh_dx, dh_dy = tape.gradient(h_pred, [x,y])
    dh_dx = tf.zeros_like(x) if dh_dx is None else dh_dx
    dh_dy = tf.zeros_like(y) if dh_dy is None else dh_dy
    
    vx = -conductivity * dh_dx
    vy = -conductivity * dh_dy
    return h_pred, vx, vy, conductivity

def generate_predictions(args):
    print("Loading trained model and generating predictions...")
    print("Configuration for prediction:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 70)

    # 1. Reconstruct the model architecture
    actual_input_dim = 2 + args.N_g + 1  # x, y, k_params (N_g), t
    h_model_instance = PirateNet(
        input_dim=actual_input_dim,
        output_dim=args.output_dim,
        m=args.piratenet_m,
        s_init_val=args.piratenet_s,
        L=args.piratenet_L,
        activation_fn_str=args.piratenet_activation,
        RWFactorized=args.piratenet_rwf
    )
    print("Model architecture reconstructed.")

    # 2. Load the trained weights
    if not os.path.exists(args.model_weights_path):
        print(f"ERROR: Model weights not found at {args.model_weights_path}")
        return
    h_model_instance.load_weights(args.model_weights_path)
    print(f"Model weights loaded from {args.model_weights_path}")

    # 3. Prepare evaluation grid and parameters
    x_eval = np.linspace(0, 1, args.eval_grid_res, dtype=np.float32)
    y_eval = np.linspace(0, 1, args.eval_grid_res, dtype=np.float32)
    X_eval_mesh, Y_eval_mesh = np.meshgrid(x_eval, y_eval)

    X_flat = X_eval_mesh.flatten()[:, None]
    Y_flat = Y_eval_mesh.flatten()[:, None]

    # k_values can be a list of lists/tuples for multiple k parameters
    # Example: [[0.5, 0.5], [0.2, 0.8]]
    # For this script, we'll process one set of k_params at a time from the command line
    k_params_eval = np.array(args.k_values, dtype=np.float32).reshape(1, args.N_g) # Ensure shape (1, N_g)
    K_params_flat = np.tile(k_params_eval, (X_flat.shape[0], 1))

    os.makedirs(args.output_data_dir, exist_ok=True)

    # Time points for evaluation
    if args.time_points:
        time_eval_points = np.array(args.time_points, dtype=np.float32)
    else: # If specific time points are not given, use a range
        time_eval_points = np.linspace(args.time_start, args.time_end, args.num_time_steps, dtype=np.float32)

    for t_val in time_eval_points:
        print(f"Generating predictions for k={args.k_values}, t={t_val:.3f}...")
        T_flat = np.full_like(X_flat, t_val, dtype=np.float32)

        # Convert to TensorFlow constants
        x_tf = tf.constant(X_flat)
        y_tf = tf.constant(Y_flat)
        k_tf = tf.constant(K_params_flat)
        t_tf = tf.constant(T_flat)

        # 4. Make predictions
        h_pred_flat, vx_pred_flat, vy_pred_flat, k_field_flat = v_model_from_h(h_model_instance, x_tf, y_tf, k_tf, t_tf)

        # 5. Reshape and save
        h_pred = tf.reshape(h_pred_flat, (args.eval_grid_res, args.eval_grid_res)).numpy()
        vx_pred = tf.reshape(vx_pred_flat, (args.eval_grid_res, args.eval_grid_res)).numpy()
        vy_pred = tf.reshape(vy_pred_flat, (args.eval_grid_res, args.eval_grid_res)).numpy()
        k_field = tf.reshape(k_field_flat, (args.eval_grid_res, args.eval_grid_res)).numpy()
        vmag_pred = np.sqrt(vx_pred**2 + vy_pred**2)

        # Sanitize k_values for filename
        k_str = "_".join([f"{k:.2f}" for k in args.k_values]).replace('.', 'p')
        output_filename = f"predictions_k_{k_str}_t_{t_val:.3f}.npz".replace('.', 'p') # Avoid dots in time for consistency
        output_path = os.path.join(args.output_data_dir, output_filename)
        
        np.savez_compressed(output_path,
                            X_mesh=X_eval_mesh, Y_mesh=Y_eval_mesh,
                            h=h_pred, Vx=vx_pred, Vy=vy_pred, Vmag=vmag_pred,
                            k_field=k_field,
                            k_params_used=args.k_values, time_point=t_val)
        print(f"Saved predictions to {output_path}")

    print("Prediction generation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply a trained Darcy Flow PINN model to generate predictions.")

    # Model architecture arguments (must match the trained model)
    parser.add_argument('--output_dim', type=int, default=1, help="Output dimension of PINN (usually 1 for head).")
    parser.add_argument('--N_g', type=int, default=2, help="Dimension of the latent k vector.")
    parser.add_argument('--piratenet_m', type=int, default=128, help="Number of RFFs (half of embedding dim).")
    parser.add_argument('--piratenet_s', type=float, default=5.0, help="Stddev for B matrix in PirateNet.")
    parser.add_argument('--piratenet_L', type=int, default=3, help="Number of residual blocks in PirateNet.")
    parser.add_argument('--piratenet_activation', type=str, default='tanh', help="Activation function for PirateNet.")
    parser.add_argument('--piratenet_rwf', type=lambda x: (str(x).lower() == 'true'), default=True, help="Use RWF layers.")

    # Model Loading
    parser.add_argument('--model_weights_path', type=str, required=True, help="Path to the trained model weights (.h5 file).")

    # Evaluation parameters
    parser.add_argument('--eval_grid_res', type=int, default=100, help="Resolution of the grid for evaluation (res x res).")
    parser.add_argument('--k_values', type=float, nargs='+', required=True, help="List of k parameter values (e.g., 0.5 0.5 for N_g=2).")
    
    # Time parameters for evaluation
    parser.add_argument('--time_points', type=float, nargs='*', default=None, help="Specific time points for evaluation (e.g., 0.1 0.5 1.0). Overrides time_start/end/steps.")
    parser.add_argument('--time_start', type=float, default=0.0, help="Start time for evaluation range (if time_points not set).")
    parser.add_argument('--time_end', type=float, default=1.0, help="End time for evaluation range (if time_points not set).")
    parser.add_argument('--num_time_steps', type=int, default=10, help="Number of time steps in range (if time_points not set).")

    # Output
    parser.add_argument('--output_data_dir', type=str, default='prediction_output', help="Directory to save prediction .npz files.")

    args = parser.parse_args()

    # Validate k_values length matches N_g
    if len(args.k_values) != args.N_g:
        parser.error(f"Number of k_values ({len(args.k_values)}) must match N_g ({args.N_g}).")

    generate_predictions(args)