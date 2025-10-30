# DarcyPINN_Project/darcy_pinn/visualize_darcy_results.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import argparse
import imageio # For creating GIFs

def plot_snapshot(data_path, output_fig_path):
    """Plots a single snapshot of h, Vmag, and quiver plot for velocity."""
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}")
        return

    X_mesh, Y_mesh = data['X_mesh'], data['Y_mesh']
    h = data['h']
    Vx = data['Vx']
    Vy = data['Vy']
    Vmag = data['Vmag']
    k_field = data['k_field'] # Assuming k_field was also saved
    k_params_used = data['k_params_used']
    time_point = data['time_point']

    k_str = "_".join([f"{k:.2f}" for k in k_params_used])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.suptitle(f'Darcy Flow Snapshot: k=({k_str}), t={time_point:.3f}', fontsize=16)

    # Plot Head (h)
    cf1 = axes[0].contourf(X_mesh, Y_mesh, h, levels=50, cmap=cm.viridis)
    fig.colorbar(cf1, ax=axes[0], label='Head (h)')
    axes[0].set_title('Hydraulic Head')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')

    # Plot Velocity Magnitude (Vmag)
    cf2 = axes[1].contourf(X_mesh, Y_mesh, Vmag, levels=50, cmap=cm.hot)
    fig.colorbar(cf2, ax=axes[1], label='Velocity Magnitude')
    # Quiver plot overlay
    stride = max(1, X_mesh.shape[0] // 20) # Adjust quiver density
    axes[1].quiver(X_mesh[::stride, ::stride], Y_mesh[::stride, ::stride],
                   Vx[::stride, ::stride], Vy[::stride, ::stride],
                   color='white', scale_units='xy', angles='xy', scale=Vmag.max()*20, # Adjust scale factor
                   headwidth=3.5, minshaft=1, width=0.003)
    axes[1].set_title('Velocity Magnitude & Vectors')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')

    # Plot Conductivity Field (k_field)
    cf3 = axes[2].contourf(X_mesh, Y_mesh, k_field, levels=50, cmap=cm.coolwarm)
    fig.colorbar(cf3, ax=axes[2], label='Conductivity a(x,y)')
    axes[2].set_title('Conductivity Field')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')

    plt.savefig(output_fig_path, dpi=300)
    print(f"Saved snapshot figure to {output_fig_path}")
    plt.close(fig)

def create_gif_from_snapshots(npz_files_dir, output_gif_path, gif_duration_per_frame):
    """Creates a GIF from a directory of snapshot .png files."""
    temp_fig_dir = os.path.join(npz_files_dir, "temp_frames_for_gif")
    os.makedirs(temp_fig_dir, exist_ok=True)
    
    images = []
    npz_files = sorted([f for f in os.listdir(npz_files_dir) if f.endswith('.npz')])
    
    if not npz_files:
        print(f"No .npz files found in {npz_files_dir} to create GIF.")
        return

    print(f"Generating frames for GIF from {len(npz_files)} .npz files...")
    for i, npz_file in enumerate(npz_files):
        data_path = os.path.join(npz_files_dir, npz_file)
        frame_path = os.path.join(temp_fig_dir, f"frame_{i:04d}.png")
        plot_snapshot(data_path, frame_path) # Re-use plot_snapshot to generate frames
        images.append(imageio.imread(frame_path))
        os.remove(frame_path) # Clean up frame
    
    if images:
        imageio.mimsave(output_gif_path, images, duration=gif_duration_per_frame, loop=0) # loop=0 for infinite loop
        print(f"Saved GIF to {output_gif_path}")
    else:
        print("No frames generated for GIF.")
    
    # Clean up temporary directory
    if os.path.exists(temp_fig_dir) and not os.listdir(temp_fig_dir): # Check if empty
        os.rmdir(temp_fig_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Darcy Flow PINN prediction results.")
    
    parser.add_argument('--input_data_path', type=str, required=True,
                        help="Path to a single .npz prediction file OR a directory containing .npz files for GIF creation.")
    parser.add_argument('--output_fig_dir', type=str, default='visualization_output',
                        help="Directory to save output figures/GIFs.")
    
    # GIF creation options (if input_data_path is a directory)
    parser.add_argument('--create_gif', action='store_true',
                        help="If set and input_data_path is a directory, create a GIF from all .npz snapshots.")
    parser.add_argument('--gif_filename', type=str, default="darcy_flow_animation.gif",
                        help="Filename for the output GIF.")
    parser.add_argument('--gif_frame_duration', type=float, default=0.2,
                        help="Duration (in seconds) for each frame in the GIF.")


    args = parser.parse_args()
    os.makedirs(args.output_fig_dir, exist_ok=True)

    if os.path.isfile(args.input_data_path) and args.input_data_path.endswith('.npz'):
        if args.create_gif:
            print("Warning: --create_gif flag is set, but --input_data_path is a single file. "
                  "Plotting single snapshot instead. For GIF, provide a directory of .npz files.")
        
        base_filename = os.path.splitext(os.path.basename(args.input_data_path))[0]
        output_png_path = os.path.join(args.output_fig_dir, f"{base_filename}_snapshot.png")
        plot_snapshot(args.input_data_path, output_png_path)

    elif os.path.isdir(args.input_data_path):
        if args.create_gif:
            output_gif_file_path = os.path.join(args.output_fig_dir, args.gif_filename)
            create_gif_from_snapshots(args.input_data_path, output_gif_file_path, args.gif_frame_duration)
        else:
            print(f"Input data path '{args.input_data_path}' is a directory. "
                  "Plotting snapshots for each .npz file found.")
            npz_files = sorted([f for f in os.listdir(args.input_data_path) if f.endswith('.npz')])
            if not npz_files:
                print(f"No .npz files found in directory: {args.input_data_path}")
            for npz_file in npz_files:
                data_path = os.path.join(args.input_data_path, npz_file)
                base_filename = os.path.splitext(npz_file)[0]
                output_png_path = os.path.join(args.output_fig_dir, f"{base_filename}_snapshot.png")
                plot_snapshot(data_path, output_png_path)
    else:
        print(f"ERROR: Invalid input_data_path: {args.input_data_path}. Must be a .npz file or a directory.")