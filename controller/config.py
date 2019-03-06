"""Configuration file for default model parameters."""

N_DEFAULT_SAMPLE = 500


def default_kwargs(X_induce, X_induce_mean=None):
    # configure estimation
    system_estimate_kwargs = dict()
    system_estimate_kwargs["step_size"] = 1e-3
    system_estimate_kwargs["max_steps"] = int(5e4)
    system_estimate_kwargs["model_dir"] = None
    system_estimate_kwargs["verbose"] = True
    system_estimate_kwargs["restart_estimate"] = False

    system_vi_kwargs = dict()
    system_vi_kwargs["vi_family"] = "dgpr"
    system_vi_kwargs["Z"] = X_induce
    system_vi_kwargs["Zm"] = X_induce_mean

    # compile overall model kwargs
    system_model_kwargs = dict(**system_estimate_kwargs,
                               **system_vi_kwargs)
    system_model_kwargs["n_sample"] = N_DEFAULT_SAMPLE

    """ 2. configure random component """
    # configure estimation
    random_estimate_kwargs = dict()
    random_estimate_kwargs["step_size"] = 1e-2
    random_estimate_kwargs["max_steps"] = int(5e4)
    random_estimate_kwargs["model_dir"] = None
    random_estimate_kwargs["verbose"] = True
    random_estimate_kwargs["restart_estimate"] = False

    # configure VI inference
    random_vi_kwargs = dict()
    random_vi_kwargs["vi_family"] = "mfvi"

    # compile overall model kwargs
    random_model_kwargs = dict(**random_estimate_kwargs,
                               **random_vi_kwargs)
    random_model_kwargs["n_sample"] = N_DEFAULT_SAMPLE

    return system_model_kwargs, random_model_kwargs
