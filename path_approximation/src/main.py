from utilities import make_log_folder

from testing_functions import run_some_linear_models,run_nn

if __name__ == '__main__':
    make_log_folder(log_folder_name="logs")
    # run_linear_model_with_under_and_over_sampling(file_name="socfb-American75", force_recreate_datasets=False, write_train_val_test=False)
    ## result: linear_regression: Accuracy=52.527%, MSE=0.492, MAE=0.558, MRE=0.2
    ##run_some_linear_models(file_name="socfb-American75", force_recreate_datasets=True, write_train_val_test=False)
    ## result: linear_regression: Accuracy = 69.712 %, MSE = 0.248, MAE = 0.393, MRE = 0.16
    
    run_nn(file_name="socfb-American75", force_recreate_datasets=True, write_train_val_test=False)
    ## result: nn: Accuracy 41.268%,  MSE = 0.225, MAE = 0.37, MRE = 0.14