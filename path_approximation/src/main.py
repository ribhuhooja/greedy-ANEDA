from utilities import make_log_folder
import shutil
from testing_functions import run_some_linear_models,run_nn
from IPython import get_ipython
import time
if __name__ == '__main__':
    make_log_folder(log_folder_name="logs")
    # run_linear_model_with_under_and_over_sampling(file_name="socfb-American75", force_recreate_datasets=False, write_train_val_test=False)
    ## result: linear_regression: Accuracy=52.527%, MSE=0.492, MAE=0.558, MRE=0.2
    #run_some_linear_models(file_name="fb-pages-food", force_recreate_datasets=True, write_train_val_test=False)
    ## result: linear_regression: Accuracy = 69.712 %, MSE = 0.248, MAE = 0.393, MRE = 0.16
    
    
    #graph = ["as-internet","socfb-American75","socfb-OR"]
    """
    run portion = 0.2, method = top_degrees, you can get many possiable results(stored in log files) like below, which doesn't make sense', 
    since all the data is the same for every run, then all the accuracy and mae should also be same
    {'acc': '6.6667', 'mse': '4.1998', 'mae': '1.6779', 'mre': '0.376'}
    {'acc': '68.7586', 'mse': '0.1565', 'mae': '0.2963', 'mre': '0.0821'}
    {'acc': '60.1372', 'mse': '0.2511', 'mae': '0.3724', 'mre': '0.0997'}
    And other tests also have the same probelms when we deal with small graphs like inf-euroroad
    method can be either random or top_degrees
    """
    graphs = ['ego-facebook-original','socfb-American75']
    portions = [0.02,0.03,0.04,0.05,0.06,0.07,0.08]
    methods = ['random','top_degrees','high_low_degrees']
    for graph_name in graphs:
        for method in methods:
            for portion in portions:
                run_nn(file_name=graph_name, force_recreate_datasets=True, portion = portion,method = method,write_train_val_test=False)
                cmd = "rm -r " + "../output/nn_return/{}".format(graph_name)
                os.system(cmd)
    
    """
    graph = ['fb-pages-food']
    portions = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    methods = ['random','top_degree']
    Acc_r = []
    Mae_r = []
    Acc_t = []
    Mae_t = []
    
    for i in range(8):
        for graph_name in graph:
            for method in methods:
                tem_acc_r = []
                tem_mae_r = []
                tem_acc_t = []
                tem_mae_t = []
                for portion in portions:
                    if(method == 'random'):
                         get_ipython().magic('reset -sf')
                         run_nn(Acc = tem_acc_r,Mae = tem_mae_r,file_name=graph_name, force_recreate_datasets=True, portion = portion,method = method,write_train_val_test=False)
                         shutil.rmtree("../output/nn_return/{}".format(graph_name))# just remove return
                         time.sleep(50)
                    else:
                         get_ipython().magic('reset -sf')
                         run_nn(Acc = tem_acc_t,Mae = tem_mae_t,file_name=graph_name, force_recreate_datasets=True, portion = portion,method = method,write_train_val_test=False)
                         shutil.rmtree("../output/nn_return/{}".format(graph_name))# just remove return
                         time.sleep(50)
                if(method == 'random'):
                    Acc_r.append(tem_acc_r)
                    Mae_r.append(tem_mae_r)
                else:
                    Acc_t.append(tem_acc_t)
                    Mae_t.append(tem_mae_t)
  
                    
    ## result: nn: Accuracy 41.268%,  MSE = 0.225, MAE = 0.37, MRE = 0.14
    
    """