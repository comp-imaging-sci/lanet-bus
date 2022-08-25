# draw bar plot with significance asterisc
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy  as np
import itertools 
import seaborn as sns
from statannotations.Annotator import Annotator
import scipy.stats as st
from matplotlib.markers import TICKDOWN

plt.rcParams.update({'font.size': 32})
plt.rc('legend', fontsize=24)
plt.rc('axes', labelsize=26)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)


NET_COLOR_PALETTE = {
    "ResNet50": "#EEFFBA", 
    "ResNet50+RMTL": "#D6FA8C", 
    "ResNet50+LA-Net":"#BEED53",  
    "ResNet50+LA-Net (no CAM)": "#5D8700", 
    "ResNet50+LA-Net (no SAM)": "#82B300", 
    "ResNet50+LA-Net (no MAM)": "#2D581A", 
    "ResNet18": "#FFDF77", 
    "ResNet18+RMTL":"#FCCF3E", 
    "ResNet18+LA-Net":"#FBB124", 
    "ResNet18+LA-Net (no CAM)":"#FD9A24", 
    "ResNet18+LA-Net (no SAM)": "#FA8223", 
    "ResNet18+LA-Net (no MAM)":"#F55F20",
    }



def flip_legned(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def stat_significance_mark(pv):
    if pv > 0.05:
        return "ns"
    elif pv <= 0.0001:
        return "****"
    elif pv <= 0.001:
        return "***"
    elif pv <= 0.01:
        return "**"
    elif pv <= 0.05:
        return "*"


def get_confidence_interval(data, metric, target, target_type="Net"):
    sample = data[metric][data[target_type] == target]
    meanv = np.mean(sample)
    std = st.sem(sample) + 1e-5
    ci = st.t.interval(alpha=0.95, df=len(sample)-1, loc=meanv, scale=std)
    return meanv/100, (ci[1] - meanv)/100 # convert to %


def get_pair_sample_p_value(data, metric, pairs):
    pvs = []
    for pair in pairs:
        a = data[metric][data["Net"] == pair[0]]
        b = data[metric][data["Net"] == pair[1]]
        pv = st.ttest_ind(a, b, alternative="less")
        pvs.append(pv[1]) # no need statistic
    return pvs


def significance_bar(ax, start, end, height, pvalue, linewidth=1.2, markersize=8, boxpad=0.15, fontsize=15, color='k', tickpad=0.03):
    # draw a line with downticks at the ends
    # pad to make the tickdown shorter is biased from the center of bars
    ax.plot([start+tickpad, end-tickpad], [height, height], '-', color=color,lw=linewidth, marker=TICKDOWN, markeredgewidth=linewidth, markersize=markersize)
    # draw the text with a bounding box covering up the line
    pmark = stat_significance_mark(pvalue)
    ax.text(0.5*(start+end), height, pmark, ha='center', va='center', bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad='+str(boxpad)), size=fontsize)


def draw_method_effect(image_name, size, dataset="covidx"):
    if dataset == "BUSI":
        csv_file = "method_effect_busi.csv"
    elif dataset == "MAYO_bbox":
        csv_file = "method_effect_mayo_bbox.csv"
    elif dataset == "ablation_mayo_bbox":
        csv_file = "ablation_mayo_bbox.csv"
    elif dataset == "ablation_busi":
        csv_file = "ablation_busi.csv" 
    elif dataset == "MAYO":
        csv_file = "method_effect_mayo.csv"
    elif dataset == "ablation_busi_iou":
        csv_file = "ablation_iou_busi.csv"
    elif dataset == "ablation_mayo_bbox_iou":
        csv_file = "ablation_iou_mayo_bbox.csv"
    df = pd.read_csv(csv_file, sep=",")

    df = df[df["Size"] == size]
    # set N groups according to the number of nets
    if dataset in ["BUSI", "MAYO_bbox"]:
        bar_width = 0.5 # bar width
        sep = 0.1       # bar separation
        sig_h = 0.025    # significance bar height
        
        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(28, 8)

        nets = ["ResNet18", "ResNet18+RMTL", "ResNet18+LA-Net", "ResNet50", "ResNet50+RMTL", "ResNet50+LA-Net"]
        pairs18 =[("ResNet18", "ResNet18+RMTL"), 
                ("ResNet18+RMTL", "ResNet18+LA-Net"), 
                ("ResNet18", "ResNet18+LA-Net")]
        pairs50 =[("ResNet50", "ResNet50+RMTL"), 
                ("ResNet50+RMTL", "ResNet50+LA-Net"), 
                ("ResNet50", "ResNet50+LA-Net")]  
        pairs = pairs18 + pairs50
        metrics = ["Precision", "Sensitivity", "Specificity", "F1-Score"]
        interval_idx = len(nets) // 2
        interval_gap = 0.4   # grouper bar group separation width
    elif dataset in ["ablation_mayo_bbox", "ablation_busi"]:
        metrics = ["Precision", "Sensitivity", "Specificity", "F1-Score"]
        nets = ["ResNet18", "ResNet18+LA-Net (no CAM)", "ResNet18+LA-Net (no SAM)", "ResNet18+LA-Net (no MAM)", "ResNet18+LA-Net", "ResNet50", "ResNet50+LA-Net (no CAM)", "ResNet50+LA-Net (no SAM)", "ResNet50+LA-Net (no MAM)", "ResNet50+LA-Net"]
        pairs18 =[("ResNet18", "ResNet18+LA-Net (no CAM)"), 
                ("ResNet18", "ResNet18+LA-Net (no SAM)"), 
                ("ResNet18", "ResNet18+LA-Net (no MAM)"),
                ("ResNet18", "ResNet18+LA-Net")]
        pairs50 =[("ResNet50", "ResNet50+LA-Net (no CAM)"), 
                ("ResNet50", "ResNet50+LA-Net (no SAM)"), 
                ("ResNet50", "ResNet50+LA-Net (no MAM)"),
                ("ResNet50", "ResNet50+LA-Net")]
        pairs = pairs18 + pairs50
        bar_width = 1.2 # bar width
        sep = 0.1       # bar separation
        sig_h = 0.025    # significance bar height
        
        interval_idx = len(nets) // 2
        interval_gap = 0.5   # grouper bar group separation width 
        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(36, 8)

    elif dataset == "MAYO":
        bar_width = 0.8 # bar width
        sep = 0.1       # bar separation
        sig_h = 0.025    # significance bar height

        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(20, 6)

        nets = ["ResNet18", "ResNet18+LA-Net", "ResNet50", "ResNet50+LA-Net"]
        pairs18 =[("ResNet18", "ResNet18+LA-Net")]
        pairs50 =[("ResNet50", "ResNet50+LA-Net")]  
        pairs = pairs18 + pairs50
        metrics = ["Precision", "Sensitivity", "Specificity", "F1-Score"]
        interval_idx = len(nets) // 2
        interval_gap = 0.4   # grouper bar group separation width 
    elif dataset in ["ablation_mayo_bbox_iou", "ablation_busi_iou"]:
        metrics = ["JSI"]
        nets = ["ResNet18+LA-Net (no CAM)", "ResNet18+LA-Net (no SAM)", "ResNet18+LA-Net (no MAM)", "ResNet18+LA-Net", "ResNet50+LA-Net (no CAM)", "ResNet50+LA-Net (no SAM)", "ResNet50+LA-Net (no MAM)", "ResNet50+LA-Net"]
        pairs18 =[("ResNet18+LA-Net (no CAM)", "ResNet18+LA-Net"), 
                ("ResNet18+LA-Net (no SAM)", "ResNet18+LA-Net"), 
                ("ResNet18+LA-Net (no MAM)", "ResNet18+LA-Net")]
        pairs50 =[("ResNet50+LA-Net (no CAM)", "ResNet50+LA-Net"), 
                ("ResNet50+LA-Net (no SAM)", "ResNet50+LA-Net"), 
                ("ResNet50+LA-Net (no MAM)", "ResNet50+LA-Net")]
        pairs = pairs18 + pairs50
        bar_width = 1 # bar width
        sep = 0.2       # bar separation
        sig_h = 0.025    # significance bar height
        
        interval_idx = len(nets) // 2
        interval_gap = 0.5   # grouper bar group separation width 
        fig, ax = plt.subplots(1, 1)
        ax = [ax]
        fig.set_size_inches(16, 10)
        # fig.set_size_inches(16, 7) 
        df[metrics[0]] = df[metrics[0]] * 100
    c = 0
    # draw bar
    for i, net in enumerate(nets):
        if i >= interval_idx:
            c = interval_gap
        for j, metric in enumerate(metrics):
            m, err = get_confidence_interval(df, metric, net)
            if err <0.0005:
                err = 0.0005
            ax[j].bar(bar_width*(i)+sep*i+c, m, width=bar_width, label=net, color=NET_COLOR_PALETTE[net], yerr=err)
    # set subplot title
    for i, metric in enumerate(metrics):
        ax[i].set_title(metric)

    # plot significance mark
    baseh = 1 
    if dataset == "MAYO":
        baseh = 0.9
    elif dataset == "ablation_mayo_bbox_iou":
        baseh = 0.75
    elif dataset == "ablation_busi_iou":
        # baseh = 0.45
        baseh = 0.6
    for i, metric in enumerate(metrics):
        pvs = get_pair_sample_p_value(df, metric, pairs)
        for j, (neta, netb), in enumerate(pairs):
            c = 0
            neta_idx = nets.index(neta)
            netb_idx = nets.index(netb) 
            if neta_idx >= interval_idx:
                c = interval_gap
            # get bar x and y 
            neta_x = bar_width*neta_idx+sep*neta_idx+c
            netb_x = bar_width*netb_idx+sep*netb_idx+c
            sig_bar_h = baseh + (netb_idx-neta_idx) * sig_h 
            # draw significance pairs 
            significance_bar(ax[i], start=neta_x, end=netb_x, height=sig_bar_h, pvalue=pvs[j], fontsize=20)

    fig_idx = list(range(len(metrics)))
    for i in fig_idx:
        if dataset in ["MAYO_bbox", "BUSI"]:
            # ax[i].set_ylim([0.75, 1.08])
            ax[i].set_ylim([0.75, 1.03]) 
            ax[i].yaxis.set_ticks(np.arange(0.75, 1.01, 0.05))
        elif dataset in ["ablation_mayo_bbox", "ablation_busi"]: 
            ax[i].set_ylim([0.8, 1.13])
            ax[i].yaxis.set_ticks(np.arange(0.8, 1.01, 0.05))
        elif dataset == "MAYO":
            ax[i].set_ylim([0.7, 1.01])
            ax[i].yaxis.set_ticks(np.arange(0.7, 1.01, 0.05))
        elif dataset == "ablation_mayo_bbox_iou":
            ax[i].set_ylim([0.5, 0.9])
            ax[i].yaxis.set_ticks(np.arange(0.5, 0.9, 0.1)) 
        elif dataset == "ablation_busi_iou":
            # ax[i].set_ylim([0.3, 0.54])
            # ax[i].yaxis.set_ticks(np.arange(0.3, 0.54, 0.05)) 
            ax[i].set_ylim([0.4, 0.69])
            ax[i].yaxis.set_ticks(np.arange(0.4, 0.69, 0.05)) 
        ax[i].set_xticks([])
    handles, labels = ax[0].get_legend_handles_labels()
    if dataset in ["MAYO_bbox", "BUSI"]:
        ax[0].legend(handles, labels, bbox_to_anchor=(0.15, -1.1, 4.4, 1), loc="upper left", ncol=6, mode="expand", borderaxespad=0, fontsize=25)
    elif dataset in ["ablation_mayo_bbox", "ablation_busi"]:
        ax[0].legend(handles, labels, bbox_to_anchor=(0.15, -1.1, 4.3, 1), loc="upper left", ncol=5, mode="expand", borderaxespad=0, fontsize=28) 
    elif dataset == "MAYO":
        ax[0].legend(handles, labels, bbox_to_anchor=(0.15, -1.1, 5, 1), loc="upper left", ncol=4, mode="expand", borderaxespad=0, fontsize=28)
    elif dataset in ["ablation_mayo_bbox_iou", "ablation_busi_iou"]:
        ax[0].legend(handles, labels, bbox_to_anchor=(0., -1.1, 1, 1), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=26)  
    fig.tight_layout()
    # plt.show()
    plt.savefig(image_name)


if __name__ == "__main__":
    # draw method 
    size = 256
    img_name = "method_effect_mayo_{}_no_sig.png".format(size)
    draw_method_effect(img_name, size=size, dataset="MAYO")

    
    # ablation
    # img_name = "ablation_mayo_bbox_iou_{}.png".format(size) 
    # draw_method_effect(img_name, size=size, dataset="ablation_mayo_bbox_iou")  

    # pv = st.ttest_ind(a, b, alternative="less")
    # print(pv)
    # sample = np.array([0.545,	0.532,	0.548])
    # meanv = np.mean(sample)
    # std = st.sem(sample)
    # ci = st.t.interval(alpha=0.95, df=len(sample)-1, loc=meanv, scale=std)
    # print(meanv, (ci[1] - meanv)) # convert to %
    