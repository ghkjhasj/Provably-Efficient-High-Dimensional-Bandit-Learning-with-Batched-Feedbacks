# Provably-Efficient-High-Dimensional-Bandit-Learning-with-Batched-Feedbacks

- For the toy simulation for the high-dimensional setting, one can run "tain.py" by choosing args.dataset='toy_lasso', args.data_dim (dimensions of the vector space) and args.sparse (the maximum number of nonzero components in the true parameter), and tuning the "arm_num" (number of arms), "batches" (number of batches), "t_0" (forced-sampling parameter), "q" (forced sample parameter for LASSO bandit), "h" (localization parameter), "lam_1" (regularization parameter) and "lam_2_0" (initial regularization parameter) accroding to the instructions in experiment section of the main paper. 

- For the warfarin task, please first download warfarin to the local device with path './data/...', and run "tain.py" by choosing args.dataset='warfarin' and changes other parameters according to the main paper.

- For the task of dynamic pricing in retail data, please fist download the dataset from the provided website to "'./data/price/...'" and then run "train.py" by choosing args.dataset='price' and changes other parameters according to the main paper.

- For the toy simulation for the low-rank setting, one can run "tain.py" by choosing args.dataset='toy_lowrank', args.d_1 (the number of columns and rows for the matrix parametr and covariates), args.rank (the maximum nnumber of ranks for the matrix parameter), and tuning the other parameters according to the details in the main paper.
