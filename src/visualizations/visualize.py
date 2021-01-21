import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#used in src/model/baseline_model.py, src/model/model.py, and src/analysis/compare.py
def visualize(name, test=False):
    
    if test:
        vis_path = 'test/'
    else:
        vis_path = 'data/visualizations/'
    
    plt.savefig(vis_path + name)
    
    return None

#used in src/model/baseline_model.py, src/model/model.py
def visualize_loss(model):
    
    plt.figure()
    plt.plot(model.history['loss'],label='Loss')
    plt.plot(model.history['val_loss'],label='Val. loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    return None
    
#used in src/model/baseline_model.py, src/model/model.py
def visualize_roc(fpr_cnn, tpr_cnn, fpr_dnn=None, tpr_dnn=None, dense=False):
    
    plt.figure()
    if dense:
        plt.plot(tpr_dnn, fpr_dnn, lw=2.5, label="Dense, AUC = {:.1f}%".format(auc(fpr_dnn,tpr_dnn)*100))
        plt.plot(tpr_cnn, fpr_cnn, lw=2.5, label="Conv1D, AUC = {:.1f}%".format(auc(fpr_cnn,tpr_cnn)*100))
    else:
        plt.plot(tpr_cnn, fpr_cnn, lw=2.5, label="Conv1D, AUC = {:.1f}%".format(auc(fpr_cnn,tpr_cnn)*100))
    plt.xlabel(r'True positive rate')
    plt.ylabel(r'False positive rate')
    plt.semilogy()
    plt.ylim(0.001,1)
    plt.xlim(0,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()
    
    return None

#used in src/analysis/compare.py
def visualize_hist(data, weight_1, weight_2, bin_vars, x_label,y_label):
    
    plt.figure()
    plt.hist(data,weights=weight_1,bins=bin_vars,density=True,alpha=0.7,label='QCD')
    plt.hist(data,weights=weight_2,bins=bin_vars,density=True,alpha=0.7,label='H(bb)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    return None

#used in src/analysis/compare.py
def visualize_roc_compare(fpr, tpr):
    
    plt.figure()
    plt.plot(fpr, tpr, lw=2.5, label="AUC = {:.1f}%".format(auc(fpr,tpr)*100))
    plt.xlabel(r'False positive rate')
    plt.ylabel(r'True positive rate')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.plot([0, 1], [0, 1], lw=2.5, label='Random, AUC = 50.0%')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()
    
    return None