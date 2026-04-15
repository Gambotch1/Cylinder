import numpy as np



config = {
    "bridge_D_ref": 7.42,  # Deck width (m)
    "cylinder_D_ref": 1.0,  # Cylinder diameter (m)
    "lookback_steps": None,  # Will be updated based on physics
    "seed": 123,  
    "motion_types": ["heave"],

    "bridge_fn_hz": 0.32,  # Natural frequency of the bridge (Hz)
    "viv_dataset": ["cylinder", "bridge"],  # Default dataset to use
    
    # Ensemble parameters
    "bridge_d_ref": 7.42,
    "bridge_downsample": 10,
    "bridge_seq_len": 900,
    "bridge_stride_train": 5,
    "bridge_t_star_release": 20.0,
    "n_ensemble": 10,
    "use_bagging": True,
    
    # Hyperparameter search ranges
    "hl_range": [50, 400],  # Hidden layer size range
    "hl_step": 10,
    "lam_range": [1e-8, 1e-3],  # Lambda range (log scale)
    
    # Optuna parameters
    "n_trials": 2,  # Reduced from 49 for Bayesian optimization
    "n_jobs": -1,  # Use all cores
    
    # Visualization
    "motion_type": "heave",
    "example_vr": 6.0, 
    "target_col": "cl",
    "input_cols": ["disp", "vel", "acc", "cl"],
    "n_models": 1,
    "n_hidden_nodes": 300,
    "alpha_reg": 1.0,
}

np.random.seed(config["seed"])