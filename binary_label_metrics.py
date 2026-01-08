"""
Module: Binary label Metrics
About:  Class for computing binay label performance metrics
"""

from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from numbers import Number
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score


class BinaryLabelMetrics:
    '''
    num_thresh - number of thresholds equally spaced in [0,1] at which the confusion matrix is computed
    '''
    def __init__(self, num_thresh=1001):
        self._numthresh = num_thresh
        self._modname,self._modname_dct = list(),dict()
        self._modname_sz = list()           #name of model with number of observations
        self._scores = list()
        self._confmat = list()
        self._auc,self._prrec = list(),list()
        self._thresh_prev = list()          #threshold at prevalence (prevalence = num ones/num obs)
        

    '''
    name - model name
    scores_df - dataframe with columns named label and score
        label: true labels with ones (events) and zeros (non-events)
        score: model output; scores in [0,1]
    params - optional dictionary of parameters
        skl_auc_average: micro, macro, weighted, samples (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
        skl_ap_average: micro, macro, weighted, samples (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
    '''
    def addModel(self, name, scores_df, params={}):
        self._modname_dct[name] = len(self._modname)
        self._modname.append(name)
        self._modname_sz.append(f"{name} ({scores_df.shape[0]:,})")
        self._scores.append(scores_df)
        
        labels,scores = scores_df["label"].values,scores_df["score"].values
        thresholds = np.linspace(max(scores.min()-1E-4,0.),min(scores.max()+1E-4,1.),self._numthresh)
        
        #confusion matrix at various thresholds
        notlab = 1-labels
        pos,neg = labels.sum(),notlab.sum()
        tp,fp,fn,tn = [np.zeros(thresholds.shape[0]+1) for _ in range(4)]
        for i in range(thresholds.shape[0]):
            tmp = (scores>=thresholds[i])*1
            tp[i] = np.einsum("i,i",labels,tmp)
            fp[i] = np.einsum("i,i",notlab,tmp)
            fn[i],tn[i] = pos-tp[i],neg-fp[i]
        
        #confusion matrix at prevalence
        inds = np.argsort(-scores)
        tp[-1],fn[-1] = labels[inds[:pos]].sum(),labels[inds[pos:]].sum()
        fp[-1],tn[-1] = pos-tp[-1],neg-fn[-1]
        thresholds = np.append(thresholds, scores[inds[pos-1]])
        self._thresh_prev.append(scores[inds[pos-1]])
        del notlab,pos,neg
        
        tot = scores_df.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            sens,spec,ppv,npv,accu,prev = tp/(tp+fn), tn/(tn+fp), tp/(tp+fp), tn/(tn+fn), (tp+tn)/tot, (tp+fn)/tot
            lift,f1 = ppv/prev,2/(1/ppv + 1/sens)
        
        df = pd.DataFrame(OrderedDict([
            ("thresh",thresholds),("tp",tp.astype(np.uint32)),("tn",tn.astype(np.uint32)),("fp",fp.astype(np.uint32)),("fn",fn.astype(np.uint32)),
                ("sens",sens),("spec",spec),("ppv",ppv),("npv",npv),("accu",accu),("prev",prev),("lift",lift),("f1",f1)   
        ]))
        df.sort_values(by="thresh", inplace=True, ignore_index=True)
        self._confmat.append(df)
        
        #ROC curve - area
        self._auc.append(roc_auc_score(labels,scores,average=params.get("skl_auc_average","micro")))
        
        #precision recall curve - average precision
        self._prrec.append(average_precision_score(labels,scores,average=params.get("skl_ap_average","micro")))


    '''
    model_names: list of model names
    return:    if model_names is empty, indices for all models are returned
               otherwise, each name is checked against the model names currently in the object and their indices are returned
    '''
    def getModelIndexes(self, model_names=[]):
        return list(range(len(self._modname)) if model_names is None or len(model_names)==0 else \
            filter(lambda x:x is not None, map(lambda x:self._modname_dct.get(x), model_names)))


    '''
    model_names:   list of model names to be plotted
    chart_types: 1=ScoreDistribtion or 2=ConfusionMatrix for different thresholds
    params - parameters used to create plots
        legloc: location of the legend (1=TR, 2=TL, 3=BL, 4=BR), can also be x,y coordinates eg (.5,.05)
    '''
    def plot(self, model_names=[], chart_types=[], params={}):
        model_idx = self.getModelIndexes(model_names)
        chart_types = [1,2,3] if chart_types is None or len(chart_types)==0 else list(filter(lambda x:x in [1,2,3], chart_types))
        save,pfx = params.get("save",False),params.get("prefix","")
        fs_ti,fs_ax,fs_le,fs_tk = 17,15,13,12
        colors = ["#F95700FF","#00A4CCFF"]
        
        def ShowOrSave(s):
            if save:
                plt.savefig(s, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        
        def ScoreDistribution(inp_df, mname):
            labels,scores = inp_df["label"].values,inp_df["score"].values
            pos,neg = scores[labels==1],scores[labels==0]
            n1,m1,s1 = len(pos),np.mean(pos),np.std(pos)
            n0,m0,s0 = len(neg),np.mean(neg),np.std(neg)
            
            bins = np.linspace(0,1,100)
            plt.figure(figsize=(12,6))
            #plt.hist(pos, bins, alpha=.65, density=True, color=colors[1], label=f"Pos {n1:>9,} ($\mu$={m1:.2f}, $\sigma$={s1:.2f})")
            #plt.hist(neg, bins, alpha=.65, density=True, color=colors[0], label=f"Neg {n0:>9,} ($\mu$={m0:.2f}, $\sigma$={s0:.2f})")
            plt.hist(pos, bins, alpha=.65, density=True, color=colors[1], label=f"Pos $\mu$={m1:.2f}, $\sigma$={s1:.2f}")
            plt.hist(neg, bins, alpha=.65, density=True, color=colors[0], label=f"Neg $\mu$={m0:.2f}, $\sigma$={s0:.2f}")
            plt.xlim([-0.01, 1.01])
            plt.xlabel("Score Bin", fontsize=fs_ax)
            plt.ylabel("Percentage of Observations Per Class", fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight="bold")
            plt.legend(loc=params.get("legloc",1), prop={"size":fs_le,"family":"monospace"})
            plt.tick_params(axis="both", which="major", labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            ShowOrSave(f"{pfx}{mname}-scores.png")
        
        def TPFPTNFN(cmat_df, mname):
            sz = .01*(cmat_df["tp"][0]+cmat_df["tn"][0]+cmat_df["fp"][0]+cmat_df["fn"][0])
            thresh,tp,tn,fp,fn = cmat_df["thresh"].values,cmat_df["tp"].values/sz,cmat_df["tn"].values/sz,cmat_df["fp"].values/sz,cmat_df["fn"].values/sz
            
            plt.figure(figsize=(12,6))
            plt.plot(thresh, tp, color=colors[1], label="TP")
            plt.plot(thresh, fp, color=colors[1], label="FP", linestyle="--")
            plt.plot(thresh, tn, color=colors[0], label="TN")
            plt.plot(thresh, fn, color=colors[0], label="FN", linestyle="--")
            plt.xlim([-0.01, 1.01])
            plt.grid(color="lightgray")
            plt.xlabel("Threshold", fontsize=fs_ax)
            plt.ylabel("Percentage of Observations", fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight="bold")
            plt.legend(loc=params.get("legloc",4), prop={"size":fs_le,"family":"monospace"})
            plt.tick_params(axis="both", which="major", labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            ShowOrSave(f"{pfx}{mname}-cmat.png")
        
        def Accuracy(cmat_df, mname):
            thresh,sens,spec,accu = cmat_df["thresh"].values,100.*cmat_df["sens"].values,100.*cmat_df["spec"].values,100.*cmat_df["accu"].values        
            
            plt.figure(figsize=(12,6))
            plt.xlim([-0.01, 1.01])
            plt.grid(color="lightgray")
            plt.plot(thresh, accu, color="black", label="accuracy")
            idx =  np.nanargmax(accu)
            plt.plot(thresh[idx], accu[idx], "x", color="black", markersize=10, zorder=200, label=f"({thresh[idx]:.2f},{accu[idx]:.1f}%)")
            plt.plot(thresh, sens, color="blue", label="sensitivity")
            plt.plot(thresh, spec, color="red", label="specificity")
            idx =  np.nanargmin(abs(sens-spec))
            plt.plot(thresh[idx], sens[idx], "o", color="magenta", markerfacecolor="none", markersize=10, zorder=100, 
                            label=f"({thresh[idx]:.2f},{sens[idx]:.1f}%)")
            plt.xlabel("Threshold", fontsize=fs_ax)
            plt.ylabel("Percentage of Observations", fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight="bold")
            plt.legend(loc=params.get("legloc",4), prop={"size":fs_le,"family":"monospace"})
            plt.tick_params(axis='both', which='major', labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,pos:'%3d%%'%x))
            ShowOrSave(f"{pfx}{mname}-accu.png")
        
        for idx in model_idx:
            if 1 in chart_types:
                ScoreDistribution(self._scores[idx], self._modname[idx])
            if 2 in chart_types:
                TPFPTNFN(self._confmat[idx], self._modname[idx])
            if 3 in chart_types:
                Accuracy(self._confmat[idx], self._modname[idx])


    '''
    model_names: list of model names to be plotted
    chart_types: 1=Receiver Operating Characteristics (ROC) or 2=Precision Recall for different thresholds
    params - parameter used to create plots
        legloc: location of the legend (1=TR, 2=TL, 3=BL, 4=BR), can also be x,y coordinates eg (.5,.05)
        save:   boolean, save chart to disk
        pfx:    prefix to filename if saved to disk, used only when save=True
        addsz:  boolean, add number of observations used to compute the AUC/AP
    '''
    def plotROC(self, model_names=[], chart_types=[], params={}):
        model_idx = self.getModelIndexes(model_names)
        chart_types = [1,2] if chart_types is None or len(chart_types)==0 else list(filter(lambda x:x in [1,2], chart_types))
        save,pfx = params.get("save",False),params.get("prefix","")
        names = self._modname_sz if params.get("addsz",True) else self._modname
        plotthresh = params.get("showthresh",[])
        
        def ROCPR(midx, ctype, labs):
            plt.figure(figsize=(8,8))
            for m in midx:
                thresh,spec,sens,ppv = self._confmat[m][["thresh","spec","sens","ppv"]].values.transpose()
                if ctype==1:
                    p = plt.plot(1-spec, sens, label=f"{names[m]} {self._auc[m]:.2%}")
                else:
                    p = plt.plot(sens, ppv, label=f"{names[m]} {self._prrec[m]:.2%}")

                for th in plotthresh:
                    idx = np.argmin(abs(thresh-th))
                    if ctype==1:
                        plt.plot(1-spec[idx], sens[idx], "o", color=p[0].get_color(), markersize=6, zorder=200)
                    else:
                        plt.plot(sens[idx], ppv[idx], "o", color=p[0].get_color(), markersize=6, zorder=200)
            
            if ctype==1:
                plt.plot([0, 1], [0, 1], color="black", linestyle=":")
            plt.xlim([-1E-2, 1.01])
            plt.ylim([-1E-2, 1.01])
            plt.grid(color="lightgray")
            plt.xlabel(labs[0], fontsize=15)
            plt.ylabel(labs[1], fontsize=15)
            plt.legend(loc=params.get("legloc",1),prop={"size":13,"family":"monospace"})
            plt.tick_params(axis="both", which="major", labelsize=12)
            
            if save:
                lbl = "roc" if ctype==1 else "pr"
                plt.savefig(f"{pfx}{lbl}.png", dpi=150, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        
        if 1 in chart_types:
            labs = ("False Positive Rate (1-Specificity)","True Positive Rate (Sensitivity)")
            ROCPR(model_idx, 1, labs)
        if 2 in chart_types:
            labs = ("Recall (Sensitivity)","Precision (Positive Predictive Value)")
            ROCPR(model_idx, 2, labs)


    '''
    model_names: list of models for which thresholds are computed
    key:        "thresh", "sens", "spec", "ppv", "npv"
    value:      floating point number; if this is empy, the confusion matrix corresponding to max value of this param is returned
    Return the confusion matrix which matches value in a key
    '''
    def confusionMatrixKeyValue(self, model_names=[], key="f1", value=None, prevalence=False):
        assert key in self._confmat[0].columns, "Error: Key not found in confustion matrix dataframe"
        model_idx = self.getModelIndexes(model_names)
        flag = True if isinstance(value,Number) else False
        
        out = pd.DataFrame()
        for m in model_idx:
            if prevalence:
                key,value,flag = "thresh",self._thresh_prev[m],True
            
            idx =  np.nanargmin(abs(self._confmat[m][key].values-value)) if flag else np.nanargmax(abs(self._confmat[m][key].values))
            out = out.append(self._confmat[m].iloc[[idx]], ignore_index=True)
        out.insert(0, "model", list(map(lambda x:self._modname[x], model_idx)))
        
        return out
        
