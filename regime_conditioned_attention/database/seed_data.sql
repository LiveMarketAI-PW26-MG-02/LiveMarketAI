USE regime_attention;
INSERT INTO experiments (run_name,d_model,n_regimes,n_sources,lr,epochs)
VALUES ("baseline_run",256,4,8,3e-4,50),("large_model",512,4,8,1e-4,80),
       ("more_regimes",256,8,8,3e-4,50),("more_sources",256,4,16,3e-4,50),
       ("low_lr",256,4,8,5e-5,100);
