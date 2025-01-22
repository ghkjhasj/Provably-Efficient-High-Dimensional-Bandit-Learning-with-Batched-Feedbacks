# Provably-Efficient-High-Dimensional-Bandit-Learning-with-Batched-Feedbacks

- Datasets Description: In the high-dimensional experiments, we simulate synthetic data based on a sparse linear model (dimensions up to 1000) and also use two real-world datasets: (1) a warfarin dosing dataset of 5,528 patients (93-dimensional features) (https://github.com/chuchro3/Warfarin/tree/master/data), and (2) a dynamic pricing dataset of 3,584 customized meal orders, each with 22 features capturing meal categories, base prices, and promotions (https://www.analyticsvidhya.com/datahack/contest/genpact-machine-learning-hackathon-1/). In the low-rank setting, we generate synthetic data under a low-rank matrix model (dimensions up to 50 and rank up to 5). All synthetic data is generated in the code.

- For the toy simulation for the high-dimensional setting, one can run "tain.py" by choosing args.dataset='toy_lasso', args.data_dim (dimensions of the vector space) and args.sparse (the maximum number of nonzero components in the true parameter), and tuning the "arm_num" (number of arms), "batches" (number of batches), "t_0" (forced-sampling parameter), "q" (forced sample parameter for LASSO bandit), "h" (localization parameter), "lam_1" (regularization parameter) and "lam_2_0" (initial regularization parameter) according to the instructions in experiment section of the main paper. 

- For the warfarin task, please first download warfarin to the local device with path './data/...', and run "tain.py" by choosing args.dataset='warfarin' and changing other parameters according to the main paper.

- For the task of dynamic pricing in retail data, please first download the dataset from the provided website to "'./data/price/...'" and then run "train.py" by choosing args.dataset='price' and changes other parameters according to the main paper.

- For the toy simulation for the low-rank setting, one can run "tain.py" by choosing args.dataset='toy_lowrank', args.d_1 (the number of columns and rows for the matrix parameter and covariates), args.rank (the maximum number of ranks for the matrix parameter), and tuning the other parameters according to the details in the main paper.
