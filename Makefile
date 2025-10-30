# DarcyPINN_Project/Makefile

SHELL:=/bin/bash
PYTHON:=python3 # Or just python, or path to your venv python

# --- Project Structure ---
PKG_NAME := darcy_pinn
TRAIN_SCRIPT := $(PKG_NAME).train_darcy_pinn
APPLY_SCRIPT := $(PKG_NAME).apply_darcy_pinn
VIS_SCRIPT   := $(PKG_NAME).visualize_darcy_results

# --- Default Directories and Names ---
OUTPUT_BASE_DIR := training_runs
PREDICTION_BASE_DIR := prediction_runs
VIS_BASE_DIR := visualization_outputs

# Default model parameters for apply/vis (MUST MATCH TRAINED MODEL)
# These are used if not overridden by specific evaluation targets
DEFAULT_N_G := 2
DEFAULT_PIRATE_M := 64
DEFAULT_PIRATE_L := 3
DEFAULT_PIRATE_S := 5.0
DEFAULT_PIRATE_ACT := tanh
DEFAULT_PIRATE_RWF := True

# --- Training Targets ---
# (Keep your existing training targets from the previous step, e.g., default_train, full_train)
# Example:
MODEL_NAME_DEFAULT := darcy_causal_piratenet_L3_m64_Ng2_default
MODEL_PATH_DEFAULT := $(OUTPUT_BASE_DIR)/$(MODEL_NAME_DEFAULT)/$(MODEL_NAME_DEFAULT)_run_final_ep*.weights.h5 
# Note: The epoch number in final model name might vary. This uses a wildcard. Be careful.

default_train:
	@echo "Running default training..."
	mkdir -p $(OUTPUT_BASE_DIR)/$(MODEL_NAME_DEFAULT)
	$(PYTHON) -m $(TRAIN_SCRIPT) \
		--output_dir $(OUTPUT_BASE_DIR)/$(MODEL_NAME_DEFAULT) \
		--model_save_name_prefix $(MODEL_NAME_DEFAULT)_run \
		--total_epochs 1000 \
		--N_chunks 2 \
		--N_g $(DEFAULT_N_G) \
		--piratenet_m $(DEFAULT_PIRATE_M) \
		--piratenet_L $(DEFAULT_PIRATE_L) \
		--log_frequency 100 \
		--save_frequency 500 \
		--seed 42

# --- Application/Inference Targets ---
# This target applies a trained model. You need to specify MODEL_WEIGHTS_PATH.
# Example: make apply_default MODEL_WEIGHTS_PATH=$(OUTPUT_BASE_DIR)/darcy_causal_piratenet_L3_m64_Ng2_default/darcy_causal_piratenet_L3_m64_Ng2_default_run_final_ep1000.weights.h5 K_VALS="0.5 0.5" T_POINTS="0.1 0.5 1.0"
PRED_OUTPUT_DIR_DEFAULT := $(PREDICTION_BASE_DIR)/$(MODEL_NAME_DEFAULT)_eval
apply_default:
	@echo "Applying default trained model..."
	@if [ -z "$(MODEL_WEIGHTS_PATH)" ]; then \
		echo "ERROR: MODEL_WEIGHTS_PATH must be set. E.g., MODEL_WEIGHTS_PATH=path/to/your/model.weights.h5"; \
		exit 1; \
	fi
	mkdir -p $(PRED_OUTPUT_DIR_DEFAULT)
	$(PYTHON) -m $(APPLY_SCRIPT) \
		--model_weights_path "$(MODEL_WEIGHTS_PATH)" \
		--N_g $(DEFAULT_N_G) \
		--piratenet_m $(DEFAULT_PIRATE_M) \
		--piratenet_L $(DEFAULT_PIRATE_L) \
		--piratenet_s $(DEFAULT_PIRATE_S) \
		--piratenet_activation $(DEFAULT_PIRATE_ACT) \
		--piratenet_rwf $(DEFAULT_PIRATE_RWF) \
		--k_values $(or $(K_VALS),0.5 0.5) \
		--time_points $(T_POINTS) \
		--output_data_dir $(PRED_OUTPUT_DIR_DEFAULT)

# --- Visualization Targets ---
# Example: make visualize_default_predictions PRED_DATA_PATH=$(PREDICTION_BASE_DIR)/darcy_causal_piratenet_L3_m64_Ng2_default_eval
VIS_OUTPUT_DIR_DEFAULT := $(VIS_BASE_DIR)/$(MODEL_NAME_DEFAULT)_plots
visualize_default_predictions:
	@echo "Visualizing predictions..."
	@if [ -z "$(PRED_DATA_PATH)" ]; then \
		echo "ERROR: PRED_DATA_PATH (directory of .npz files) must be set."; \
		exit 1; \
	fi
	mkdir -p $(VIS_OUTPUT_DIR_DEFAULT)
	$(PYTHON) -m $(VIS_SCRIPT) \
		--input_data_path "$(PRED_DATA_PATH)" \
		--output_fig_dir $(VIS_OUTPUT_DIR_DEFAULT)

# Example: make gif_default_predictions PRED_DATA_PATH=...
gif_default_predictions:
	@echo "Creating GIF from predictions..."
	@if [ -z "$(PRED_DATA_PATH)" ]; then \
		echo "ERROR: PRED_DATA_PATH (directory of .npz files) must be set."; \
		exit 1; \
	fi
	mkdir -p $(VIS_OUTPUT_DIR_DEFAULT)
	$(PYTHON) -m $(VIS_SCRIPT) \
		--input_data_path "$(PRED_DATA_PATH)" \
		--output_fig_dir $(VIS_OUTPUT_DIR_DEFAULT) \
		--create_gif \
		--gif_filename $(MODEL_NAME_DEFAULT)_animation.gif \
		--gif_frame_duration 0.15


.PHONY: default_train apply_default visualize_default_predictions gif_default_predictions clean

clean:
	@echo "Cleaning up output directories..."
	rm -rf $(OUTPUT_BASE_DIR)/$(MODEL_NAME_DEFAULT)
	rm -rf $(PREDICTION_BASE_DIR)/$(MODEL_NAME_DEFAULT)_eval
	rm -rf $(VIS_BASE_DIR)/$(MODEL_NAME_DEFAULT)_plots
	@echo "Cleanup complete."