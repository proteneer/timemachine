def test_smc_freesolv_fit(smc_free_solv_path):
    """
    refit_freesolv given the free solv results on two molecules.
    Expect the fit MAE to be low, since we are fitting on all of the data.
    Also test the loss-only mode which should give similar results.
    """
    dG_preds, dG_expts = get_smc_free_solv_results(smc_free_solv_path)
    mean_abs_err_kcalmol_orig = np.mean(np.abs(dG_preds - dG_expts))

    def read_dgs():
        dG_preds = []
        dG_expts = []
        with open("fit_pred_dg.csv", "r") as f:
            for row in csv.DictReader(f):
                # Convert to kcal/mol to be consistent with above test
                dG_preds.append(float(row["pred_dg (kJ/mol)"]) / KCAL_TO_KJ)
                dG_expts.append(float(row["exp_dg (kJ/mol)"]) / KCAL_TO_KJ)
        dG_preds = np.array(dG_preds)
        dG_expts = np.array(dG_expts)
        return dG_preds, dG_expts

    with temporary_working_dir() as temp_dir:
        config = dict(result_path=smc_free_solv_path, n_mols=2, n_gpus=1)
        run_example("refit_freesolv.py", get_cli_args(config), cwd=temp_dir)
        dG_preds, dG_expts = read_dgs()

        fit_ff_fname = "fit_ffld_all_final.py"
        assert Path(fit_ff_fname).exists()

        config = dict(result_path=smc_free_solv_path, loss_only=None, ff_refit=fit_ff_fname, n_mols=2, n_gpus=1)
        run_example("refit_freesolv.py", get_cli_args(config), cwd=temp_dir)
        dG_preds_lo, dG_expts_lo = read_dgs()

    mean_abs_err_kcalmol_fit = np.mean(np.abs(dG_preds - dG_expts))
    assert mean_abs_err_kcalmol_fit < mean_abs_err_kcalmol_orig
    assert mean_abs_err_kcalmol_fit < 0.1

    mean_abs_err_kcalmol_lo = np.mean(np.abs(dG_preds_lo - dG_expts_lo))
    assert mean_abs_err_kcalmol_fit == pytest.approx(mean_abs_err_kcalmol_lo, abs=1e-1)
