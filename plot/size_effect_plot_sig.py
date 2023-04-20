# draw bar plot with significance asterisc
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy  as np
import itertools 
import seaborn as sns
from statannotations.Annotator import Annotator
import scipy.stats as st
from matplotlib.markers import TICKDOWN
import matplotlib.colors as mcolors
import re

plt.rcParams.update({'font.size': 40})
plt.rc('legend', fontsize=24)
# plt.rc('axes', labelsize=26)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=30)
plt.rc('axes', labelsize=40) 
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)

mat_colors = mcolors.TABLEAU_COLORS

NET_COLOR_PALETTE = {
    # "R50": "#EEFFBA", 
    # "R50+RMTL": "#D6FA8C", 
    # "R50+LA-Net":"#BEED53",
    "R50": mat_colors["tab:blue"],
    "R50+RMTL (1)": mat_colors["tab:orange"], 
    "R50+LA-Net (1)": mat_colors["tab:green"],
    "R50+LA-Net-SSL": mat_colors["tab:green"],
    "R50+LA-Net-FSL": mat_colors["tab:green"],
    "R50+RMTL": mat_colors["tab:orange"],
    "R50+RMTL-SSL": mat_colors["tab:orange"], 
    "R50+RMTL-FSL": mat_colors["tab:orange"],  
    "R50+LA-Net": mat_colors["tab:green"],
    "R50+LA-Net-256": mat_colors["tab:green"],
    "R50+LA-Net-512": mat_colors["tab:green"],
    "R50+LA-Net (no CAM)": "#5D8700", 
    "R50+LA-Net (no SAM)": "#82B300", 
    "R50+LA-Net (no MAM)": "#2D581A", 
    # "R18": "#FFDF77", 
    # "R18+RMTL":"#FCCF3E", 
    # "R18+LA-Net":"#FBB124", 
    "R18": mat_colors["tab:red"],
    "R18+RMTL (1)": mat_colors["tab:purple"],
    "R18+LA-Net (1)": mat_colors["tab:brown"], 
    "R18+LA-Net-SSL": mat_colors["tab:brown"],
    "R18+LA-Net-FSL": mat_colors["tab:brown"],
    "R18+RMTL": mat_colors["tab:purple"],
    "R18+RMTL-SSL": mat_colors["tab:purple"],
    "R18+RMTL-FSL": mat_colors["tab:purple"],
    "R18+LA-Net": mat_colors["tab:brown"],
    "R18+LA-Net-256": mat_colors["tab:brown"],  
    "R18+LA-Net-512": mat_colors["tab:brown"], 
    "R18+LA-Net (no CAM)":"#FD9A24", 
    "R18+LA-Net (no SAM)": "#FA8223", 
    "R18+LA-Net (no MAM)":"#F55F20",
    "ViT": "#B03A2E",
    "EB0": "#196F3D",
    "UNet (1)": "#2E4053",
    "UNet": "#2E4053",
    "AGN (1)": mat_colors["tab:pink"], 
    "AGN": mat_colors["tab:pink"], 
    "DeepLabV3": mat_colors["tab:cyan"],
    "DeepLabV3 (1)": mat_colors["tab:cyan"], 
    }

PART_DATA_PALETTE = {
    # "R50+LA-Net": "#82B300", 
    # "R18+LA-Net": "#FBB124",
    "R18": mat_colors["tab:red"],
    "R50": mat_colors["tab:blue"],
    "R18+RMTL" : mat_colors["tab:purple"],
    "R18+LA-Net": mat_colors["tab:brown"],
    "R50+RMTL": mat_colors["tab:orange"], 
    "R50+LA-Net": mat_colors["tab:green"],
    "EB0": mat_colors["tab:olive"],
    "UNet": mat_colors["tab:cyan"],
}

def flip_legned(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def stat_significance_mark(pv):
    if pv > 0.05:
        return "ns"
    # elif pv <= 0.0001:
    #     return "****"
    elif pv <= 0.001:
        return "***"
    elif pv <= 0.01:
        return "**"
    elif pv <= 0.05:
        return "*"


def get_confidence_interval(data, metric, target, target_type="Net"):
    sample = data[metric][data[target_type] == target].astype(float)
    meanv = np.mean(sample)
    std = st.sem(sample) + 1e-5
    ci = st.t.interval(alpha=0.95, df=len(sample)-1, loc=meanv, scale=std)
    # return meanv/100, (ci[1] - meanv)/100 # convert to %
    return meanv, (ci[1] - meanv) 


def get_pair_sample_p_value(data, metric, pairs):
    pvs = []
    for pair in pairs:
        a = data[metric][data["Net"] == pair[0]].astype(float)
        b = data[metric][data["Net"] == pair[1]].astype(float)
        pv = st.ttest_ind(a, b, alternative="less")
        pvs.append(pv[1]) # no need statistic
    return pvs


def significance_bar(ax, start, end, height, pvalue, linewidth=1.2, markersize=8, boxpad=0.15, fontsize=15, color='k', tickpad=0.03):
    # draw a line with downticks at the ends
    # pad to make the tickdown shorter is biased from the center of bars
    ax.plot([start+tickpad, end-tickpad], [height, height], '-', color=color,lw=linewidth, marker=TICKDOWN, markeredgewidth=linewidth, markersize=markersize)
    # draw the text with a bounding box covering up the line
    pmark = stat_significance_mark(pvalue)
    ax.text(0.5*(start+end), height-0.002, pmark, ha='center', va='center', bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad='+str(boxpad)), size=fontsize)


def draw_method_effect(image_name, size, dataset):
    if dataset in "busi_cls":
        csv_file = "method_effect_busi.csv"
    elif dataset == "mayo_bbox_cls":
        csv_file = "method_effect_mayo_bbox.csv"
    elif dataset == "mayo_cls":
        csv_file = "method_effect_mayo.csv"
    elif dataset == "busi_iou":
        csv_file = "ablation_busi_datasize_iou.csv"
    elif dataset == "mayo_bbox_iou":
        csv_file = "method_effect_mayo_bbox_iou.csv"
    elif dataset == "mayo_iou":
        csv_file = "method_effect_mayo_iou.csv"
    elif dataset == "ablation_busi_iou":
        csv_file = "ablation_busi_att_iou.csv"
    elif dataset == "ablation_mayo_bbox_iou":
        csv_file = "ablation_mayo_bbox_att_iou.csv"
    elif dataset == "ablation_mayo_bbox_att":
        csv_file = "ablation_mayo_bbox_att.csv"
    elif dataset == "ablation_busi_att":
        csv_file = "ablation_busi_att.csv" 
    elif dataset == "ablation_mayo_att":
        csv_file = "ablation_mayo_att.csv" 
    elif dataset == "ablation_mayo_att_iou":
        csv_file = "ablation_mayo_att_iou.csv"  
    elif dataset == "ablation_busi_net_depth":
        csv_file = "ablation_busi_net_depth.csv"
    elif dataset == "ablation_mayo_bbox_net_depth":
        csv_file = "ablation_mayo_bbox_net_depth.csv"
    elif dataset == "ablation_mayo_net_depth":
        csv_file = "ablation_mayo_net_depth.csv"
    elif dataset == "ablation_busi_net_depth_iou":
        csv_file = "ablation_busi_net_depth_iou.csv"
    elif dataset == "ablation_mayo_bbox_net_depth_iou":
        csv_file = "ablation_mayo_bbox_net_depth_iou.csv"
    elif dataset == "ablation_mayo_net_depth_iou":
        csv_file = "method_effect_mayo_iou.csv" 
    elif dataset == "ablation_busi_size":
        csv_file = "ablation_busi_size.csv"
    elif dataset == "ablation_mayo_bbox_size":
        csv_file = "ablation_mayo_bbox_size.csv"
    elif dataset == "ablation_mayo_size":
        csv_file = "ablation_mayo_size.csv"
    elif dataset == "ablation_mayo_semi":
        csv_file = "ablation_mayo_semi.csv"
    elif dataset == "ablation_mayo_semi_iou":
        csv_file = "ablation_mayo_semi_iou.csv"
    df = pd.read_csv(csv_file, sep=",")

    if not re.search("(size|semi)", dataset): 
        df = df[df["Size"] == size]
    # set N groups according to the number of nets
    if dataset in ["busi_cls", "mayo_bbox_cls", "mayo_cls"]:
        metrics = ["Precision", "Recall", "Specificity", "F1-Score", "AUC", "Accuracy"]
        nets = ["ViT", "EB0", "AGN", "R18",   "R50", "R18+RMTL", "R50+RMTL", "R18+LA-Net", "R50+LA-Net"]
        pairs18 =[("EB0", "R18+LA-Net"),
                ("ViT", "R18+LA-Net"),
                ("AGN", "R18+LA-Net"), 
                ("R18", "R18+LA-Net"),
                ("R50", "R18+LA-Net"),
                ("R18+RMTL", "R18+LA-Net"),
                ("R50+RMTL", "R18+LA-Net"),  
                ]
        pairs50 =[ ("EB0", "R50+LA-Net"),
                ("ViT", "R50+LA-Net"),
                ("AGN", "R50+LA-Net"), 
                ("R18", "R50+LA-Net"),
                ("R50", "R50+LA-Net"),
                ("R18+RMTL", "R50+LA-Net"), 
                ("R50+RMTL", "R50+LA-Net"), 
                ] 
        pairs18 = [] 
        pairs = pairs18 + pairs50
        df[metrics] = df[metrics] / 100
    elif dataset in ["ablation_mayo_bbox_att", "ablation_busi_att", "ablation_mayo_att"]:
        metrics = ["Precision", "Recall", "Specificity", "F1-Score", "AUC", "Accuracy"]
        # nets = ["R18", "R18+LA-Net (no CAM)", "R18+LA-Net (no SAM)", "R18+LA-Net (no MAM)", "R18+LA-Net", "R50", "R50+LA-Net (no CAM)", "R50+LA-Net (no SAM)", "R50+LA-Net (no MAM)", "R50+LA-Net"]
        nets = ["R50", "R50+LA-Net (no CAM)", "R50+LA-Net (no SAM)", "R50+LA-Net (no MAM)", "R50+LA-Net"]
        pairs18 =[("R18+LA-Net (no CAM)", "R18+LA-Net"), 
                ("R18+LA-Net (no SAM)", "R18+LA-Net"), 
                ("R18+LA-Net (no MAM)", "R18+LA-Net"),
                ("R18", "R18+LA-Net")]
        pairs50 =[("R50+LA-Net (no CAM)", "R50+LA-Net"), 
                ("R50+LA-Net (no SAM)", "R50+LA-Net"), 
                ("R50+LA-Net (no MAM)", "R50+LA-Net"),
                ("R50", "R50+LA-Net")]
        pairs18 = []
        pairs = pairs18 + pairs50
        df[metrics] = df[metrics] / 100
    elif dataset in ["ablation_mayo_bbox_att_iou", "ablation_busi_att_iou", "ablation_mayo_att_iou"]:
        metrics = ["IoU", "Dice"]
        # nets = ["R18+LA-Net (no CAM)", "R18+LA-Net (no SAM)", "R18+LA-Net (no MAM)", "R18+LA-Net", "R50+LA-Net (no CAM)", "R50+LA-Net (no SAM)", "R50+LA-Net (no MAM)", "R50+LA-Net"]
        nets = ["R50", "R50+LA-Net (no CAM)", "R50+LA-Net (no SAM)", "R50+LA-Net (no MAM)", "R50+LA-Net"]
        pairs18 =[("R18+LA-Net (no CAM)", "R18+LA-Net"), 
                ("R18+LA-Net (no SAM)", "R18+LA-Net"), 
                ("R18+LA-Net (no MAM)", "R18+LA-Net")]
        pairs50 =[("R50+LA-Net (no CAM)", "R50+LA-Net"), 
                ("R50+LA-Net (no SAM)", "R50+LA-Net"), 
                ("R50+LA-Net (no MAM)", "R50+LA-Net")]
        pairs18 = []
        pairs = pairs18 + pairs50
        # df[metrics] = df[metrics] * 100
    elif dataset in ["busi_iou", "mayo_bbox_iou", "mayo_iou"]:
        metrics = ["IoU", "Dice"]
        # nets = ["AGN (1)", "UNet (1)", "DeepLabV3 (1)", "R18+RMTL (1)", "R50+RMTL (1)", "R18+LA-Net (1)", "R50+LA-Net (1)",]
        # pairs18 =[("AGN (1)", "R18+LA-Net (1)"),
        #         ("UNet (1)", "R18+LA-Net (1)"), 
        #         ("DeepLabV3 (1)", "R18+LA-Net (1)"),
        #         ("R18+RMTL (1)", "R18+LA-Net (1)"),
        #         ("R50+RMTL (1)", "R18+LA-Net (1)") 
        #         ]
        # pairs18 = []
        # pairs50 =[
        #         ("AGN (1)", "R50+LA-Net (1)"),
        #         ("UNet (1)", "R50+LA-Net (1)"), 
        #         ("DeepLabV3 (1)", "R50+LA-Net (1)"),
        #         ("R50+RMTL (1)", "R50+LA-Net (1)"), 
        #         ("R18+RMTL (1)", "R50+LA-Net (1)")]
        nets = ["AGN", "UNet", "DeepLabV3", "R18+RMTL", "R18+LA-Net", "R50+RMTL", "R50+LA-Net",]
        pairs18 =[ ("AGN", "R18+LA-Net"),
                ("UNet", "R18+LA-Net"), 
                ("DeepLabV3", "R18+LA-Net"),
                ("R18+RMTL", "R18+LA-Net"),
                ("R50+RMTL", "R18+LA-Net") 
                ]
        
        pairs50 =[("AGN", "R50+LA-Net"),
                ("UNet", "R50+LA-Net"), 
                ("DeepLabV3", "R50+LA-Net"),
                ("R50+RMTL", "R50+LA-Net"), 
                ("R18+RMTL", "R50+LA-Net")]
        pairs50 = []
        pairs = pairs18 + pairs50
        # df[metrics] = df[metrics] * 100
    elif dataset in ["ablation_busi_net_depth", "ablation_mayo_bbox_net_depth", "ablation_mayo_net_depth",]:
        nets = ["R18", "R18+RMTL", "R18+LA-Net", "R50", "R50+RMTL", "R50+LA-Net"]
        pairs18 =[ ("R18", "R18+LA-Net"),
                ("R18+RMTL", "R18+LA-Net"), 
                ]
        pairs50 =[("R50", "R50+LA-Net"),
                ("R50+RMTL", "R50+LA-Net"), ] 
        pairs = pairs18 + pairs50
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC",]
        df[metrics] = df[metrics] / 100 
    elif dataset in ["ablation_busi_net_depth_iou", "ablation_mayo_bbox_net_depth_iou", "ablation_mayo_net_depth_iou"]:
        metrics = ["IoU", "Dice"]
        nets = ["R18+RMTL", "R18+LA-Net", "R50+RMTL",  "R50+LA-Net",]
        pairs18 =[
                ("R18+RMTL", "R18+LA-Net"),
                ]
        pairs50 =[ 
                ("R50+RMTL", "R50+LA-Net")]
        pairs = pairs18 + pairs50

    elif dataset in ["ablation_busi_size", "ablation_mayo_bbox_size", "ablation_mayo_size"]:
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        # metrics =  ["IoU", "Dice"]
        nets = ["R18+LA-Net-256",  "R18+LA-Net-512", "R50+LA-Net-256", "R50+LA-Net-512"]

        pairs = [("R18+LA-Net-256", "R18+LA-Net-512"), ("R50+LA-Net-256", "R50+LA-Net-512")]
        # pairs = []
        df[metrics] = df[metrics] / 100
    elif dataset in ["ablation_busi_size_iou", "ablation_mayo_bbox_size_iou", "ablation_mayo_size_iou"]:
        metrics = ["IoU", "Dice"]
        nets = ["R18+LA-Net-256",  "R18+LA-Net-512", "R50+LA-Net-256", "R50+LA-Net-512"]
        pairs = [("R18+LA-Net-256", "R18+LA-Net-512"), ("R50+LA-Net-256", "R50+LA-Net-512")]
        # pairs = []
        
    elif dataset == "ablation_mayo_semi":
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        nets = [  "R18+RMTL-FSL", "R18+RMTL-SSL", "R50+RMTL-FSL", "R50+RMTL-SSL", "R18+LA-Net-FSL", "R18+LA-Net-SSL",  "R50+LA-Net-FSL", "R50+LA-Net-SSL", ]
        pairs18 =[
                ("R18+RMTL-FSL", "R18+RMTL-SSL"),
                ("R18+LA-Net-FSL", "R18+LA-Net-SSL"),
                ]

        pairs50 =[ 
                ("R50+RMTL-FSL", "R50+RMTL-SSL"),
                ("R50+LA-Net-FSL", "R50+LA-Net-SSL"),
                ]
        pairs = pairs18 + pairs50
    
        df[metrics] = df[metrics] / 100
    elif dataset == "ablation_mayo_semi_iou":
        metrics = ["IoU", "Dice"]
        nets = [  "R18+RMTL-FSL", "R18+RMTL-SSL", "R50+RMTL-FSL", "R50+RMTL-SSL", "R18+LA-Net-FSL", "R18+LA-Net-SSL",  "R50+LA-Net-FSL", "R50+LA-Net-SSL", ]
        pairs18 =[
                ("R18+RMTL-FSL", "R18+RMTL-SSL"),
                ("R18+LA-Net-FSL", "R18+LA-Net-SSL"),
                ]

        pairs50 =[ 
                ("R50+RMTL-FSL", "R50+RMTL-SSL"),
                ("R50+LA-Net-FSL", "R50+LA-Net-SSL"),
                ]
        pairs = pairs18 + pairs50
    

    # set bar or box plot params
    bar_width = 0.5 # bar width
    sep = 0.1       # bar separation
    sig_h = 0.025    # significance bar height
    
    fig, ax = plt.subplots(1, len(metrics))
    fig.set_size_inches(7*len(metrics), 8)
    interval_idx = len(nets) // 2
    interval_gap = 0.4   # grouper bar group separation width

    # print mean and error 
    for i, net in enumerate(nets):
        for j, metric in enumerate(metrics):
            m, err = get_confidence_interval(df, metric, net)
            print(net, metric, "${:.3f}\pm{:.3f}$".format(m, err))

    # draw bar
    # for i, net in enumerate(nets):
    #     if i >= interval_idx:
    #         c = interval_gap
    #     else:
    #         c = 0
    #     # data = []
    #     for j, metric in enumerate(metrics):
    #         m, err = get_confidence_interval(df, metric, net)
    #         print(net, metric, "${:.3f}\pm{:.3f}$".format(m, err))
    #         sample = df[metric][df["Net"] == net].astype(float)
    #         # data.append(sample)
    #         if err <0.0005:
    #             err = 0.0005
    #         ax[j].bar(bar_width*(i)+sep*i+c, m, width=bar_width, label=net, yerr=err, color=NET_COLOR_PALETTE[net])


    # draw box plot
    for i, metric in enumerate(metrics):
        data = []
        pos = []
        widths = []
        colors = []
        for j, net in enumerate(nets):
            if j >= interval_idx:
                c = interval_gap 
            else:
                c = 0
            sample = df[metric][df["Net"] == net].astype(float) 
            data.append(sample)
            x = bar_width*j+sep*j+c
            pos.append(x)
            widths.append(bar_width)
            colors.append(NET_COLOR_PALETTE[net])
        bp = ax[i].boxplot(data, positions=pos, widths=widths, labels=nets, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set(edgecolor=None)
        for median in bp["medians"]:
            median.set(color='black')

    # # set subplot title
    for i, metric in enumerate(metrics):
        ax[i].set_title(metric)

    # draw p-value first
    # plot significance mark
    # baseh = 1 
    # if dataset == "mayo_cls":
    #     baseh = 0.9
    # elif dataset == "ablation_mayo_bbox_att_iou":
    #     baseh = 0.75
    # elif dataset == "ablation_busi_att_iou":
    #     baseh = 0.45
    #     # baseh = 0.6
    # elif dataset in ["busi_iou", "mayo_bbox_iou"]:
    #     baseh = 0.6
    # elif dataset in ["ablation_busi_net_depth_iou", "ablation_mayo_bbox_net_depth_iou", "ablation_busi_size_iou", "ablation_mayo_bbox_size_iou"]:
    #     baseh = 0.75
    # elif dataset in ["ablation_busi_size", "ablation_mayo_bbox_size"]:
    #     baseh = 0.98
    for i, metric in enumerate(metrics):
        pvs = get_pair_sample_p_value(df, metric, pairs)
        for j, (neta, netb), in enumerate(pairs):
            c = 0
            neta_idx = nets.index(neta)
            netb_idx = nets.index(netb) 
            if neta_idx >= interval_idx:
                c = interval_gap
            # get bar x and y 
            baseh = max(np.max(df[metric][df["Net"]==neta]), np.max(df[metric][df["Net"]==netb])) 
            neta_x = bar_width*neta_idx+sep*neta_idx+c
            netb_x = bar_width*netb_idx+sep*netb_idx+c
            sig_bar_h = baseh + (netb_idx-neta_idx) * sig_h 
            # draw significance pairs 
            significance_bar(ax[i], start=neta_x, end=netb_x, height=sig_bar_h, pvalue=pvs[j], fontsize=30)

    fig_idx = list(range(len(metrics)))
    for i in fig_idx:
        # if dataset in ["mayo_bbox_cls", "busi_cls", "mayo_cls"]:
        #     # ax[i].set_ylim([0.75, 1.08])
        #     ax[i].set_ylim([0.75, 1.03]) 
        #     ax[i].yaxis.set_ticks(np.arange(0.75, 1.01, 0.05))
        # elif dataset in ["ablation_busi_net_depth", "ablation_mayo_bbox_net_depth"]:
        #     ax[i].set_ylim([0.8, 1.03]) 
        #     ax[i].yaxis.set_ticks(np.arange(0.8, 1.01, 0.05))
        # elif dataset in ["ablation_busi_size", "ablation_mayo_bbox_size"]:
        #     ax[i].set_ylim([0.9, 1.01]) 
        #     ax[i].yaxis.set_ticks(np.arange(0.9, 1.01, 0.02))
        # elif dataset in ["ablation_mayo_bbox_att", "ablation_busi_att"]: 
        #     ax[i].set_ylim([0.8, 1.13])
        #     ax[i].yaxis.set_ticks(np.arange(0.8, 1.01, 0.05))
        # elif dataset in ["ablation_busi_att_iou", "busi_iou", "mayo_bbox_iou", "ablation_mayo_bbox_att_iou"]:
            # ax[i].set_ylim([0.3, 0.54])
            # ax[i].yaxis.set_ticks(np.arange(0.3, 0.54, 0.05)) 
        #     ax[i].set_ylim([0.3, 0.8])
        #     ax[i].yaxis.set_ticks(np.arange(0.3, 0.8, 0.05)) 
        # elif dataset in ["ablation_busi_net_depth_iou", "ablation_mayo_bbox_net_depth_iou"]:
        #     ax[i].set_ylim([0.35, 0.63])
        #     ax[i].yaxis.set_ticks(np.arange(0.3, 0.61, 0.05)) 
        # elif dataset in ["ablation_busi_size_iou", "ablation_mayo_bbox_size_iou"]:
        #     ax[i].set_ylim([0.4, 0.63])
        #     ax[i].yaxis.set_ticks(np.arange(0.4, 0.61, 0.05)) 
        ax[i].set_xticks([])
        if re.search("size", dataset):
            # ax[i].set_xticks([(pos[0]+pos[1])/2, (pos[2]+pos[3])/2])
            # ax[i].set_xticklabels(["256 vs 512", "256 vs 512"])
            ax[i].set_xticks(pos)
            ax[i].set_xticklabels(["256", "512", "256", "512"])
        elif re.search("semi", dataset):
            ax[i].set_xticks([(pos[0]+pos[1])/2, (pos[2]+pos[3])/2])
            ax[i].set_xticklabels(["FSL", "SSL"])
    # handles, labels = ax[0].get_legend_handles_labels()
    handles, labels = bp["boxes"], nets
    if re.search("size", dataset): 
        handles, labels = [bp["boxes"][0], bp["boxes"][2]], ["R18+LA-Net", "R50+LA-Net"]
        # handles, labels = [bp["boxes"][0], bp["boxes"][1]], ["R18+LA-Net", "R50+LA-Net"]
    elif re.search("semi", dataset):
        handles, labels = [bp["boxes"][0], bp["boxes"][2], bp["boxes"][4], bp["boxes"][6]],  [  "R18+RMTL", "R50+RMTL", "R18+LA-Net",  "R50+LA-Net", ]
    if dataset in ["mayo_bbox_cls", "busi_cls", "mayo_cls"]:
        ax[0].legend(handles, labels, bbox_to_anchor=(0.15, -1.1, 4.4, 1), loc="upper left", ncol=6, mode="expand", borderaxespad=0, fontsize=25)
    elif dataset in ["ablation_mayo_bbox_att", "ablation_busi_att"]:
        ax[0].legend(handles, labels, bbox_to_anchor=(0.15, -1.1, 4.3, 1), loc="upper left", ncol=5, mode="expand", borderaxespad=0, fontsize=28) 
    elif dataset in ["ablation_mayo_bbox_att_iou", "ablation_busi_att_iou", "busi_iou", "mayo_bbox_iou", "mayo_iou"]:
        # ax[0].legend(handles, labels, bbox_to_anchor=(0., -1.1, 1, 1), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=26)
        ax[0].legend(handles, labels, bbox_to_anchor=(0., -1.0, 2, 1), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=26)  
    elif dataset in ["ablation_busi_net_depth", "ablation_mayo_bbox_net_depth", "ablation_mayo_net_depth", "ablation_mayo_semi"]:
        ax[0].legend(handles, labels, bbox_to_anchor=(0.5, -1.1, 5.7, 1), loc="upper left", ncol=6, mode="expand", borderaxespad=0, fontsize=40)
    elif dataset in ["ablation_busi_net_depth_iou", "ablation_mayo_bbox_net_depth_iou", "ablation_busi_size_iou", "ablation_mayo_bbox_size_iou", "ablation_mayo_net_depth_iou"]:
        # ax[0].legend(handles, labels, bbox_to_anchor=(0., -1.1, 1, 1), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=26)
        ax[0].legend(handles, labels, bbox_to_anchor=(0.2, -1.1, 2., 1), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=32) 
    elif dataset in ["ablation_busi_size", "ablation_mayo_bbox_size", "ablation_mayo_size", "ablation_mayo_semi", "ablation_mayo_semi_iou"]:
        ax[1].legend(handles, labels, bbox_to_anchor=(0.5, -1.2, 2.8, 1), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=32)
        # ax[0].legend(handles, labels, bbox_to_anchor=(0.3, -1.2, 1.8, 1), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=30)   
    fig.tight_layout()
    # plt.show()
    plt.savefig(image_name)


def draw_confidence_band(image_name, size, type):
    if type == "part_data":
        csv_file = "ablation_busi_datasize.csv"
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        nets = [ "R18+RMTL (0.25)", "R18+RMTL (0.5)", "R18+RMTL (0.75)", "R18+RMTL (1)", "R50+RMTL (0.25)", "R50+RMTL (0.5)", "R50+RMTL (0.75)", "R50+RMTL (1)",  "R50+LA-Net (0.25)", "R50+LA-Net (0.5)", "R50+LA-Net (0.75)", "R50+LA-Net (1)", ]
        # nets = ["R18 (0.25)", "R18 (0.5)", "R18 (0.75)", "R18 (1)", "R50 (0.25)", "R50 (0.5)", "R50 (0.75)", "R50 (1)", "EB0 (0.25)", "EB0 (0.5)", "EB0 (0.75)", "EB0 (1)",   "R18+RMTL (0.25)", "R18+RMTL (0.5)", "R18+RMTL (0.75)", "R18+RMTL (1)", "R50+RMTL (0.25)", "R50+RMTL (0.5)", "R50+RMTL (0.75)", "R50+RMTL (1)",  "R50+LA-Net (0.25)", "R50+LA-Net (0.5)", "R50+LA-Net (0.75)", "R50+LA-Net (1)", ]

    
        # "R18+LA-Net (0.25)", "R18+LA-Net (0.5)", "R18+LA-Net (0.75)", "R18+LA-Net (1)",
        draw_nets = ["R18+RMTL", "R50+RMTL",  "R50+LA-Net"]
        # draw_nets = ["R18", "R50", "EB0", "R18+RMTL", "R50+RMTL",  "R50+LA-Net"]
        targets = ["0.25", "0.5", "0.75", "1"]
        # markers = ["o", "v"] 
        markers = ["x"] * len(draw_nets)
        

        # fig.set_size_inches(22, 6)   # no legend
    elif type == "part_data_iou":
        csv_file = "ablation_busi_datasize_iou.csv"
        metrics = ["IoU", "Dice"]
        nets = ["UNet (0.25)", "UNet (0.5)", "UNet (0.75)", "UNet (1)", "R18+RMTL (0.25)", "R18+RMTL (0.5)", "R18+RMTL (0.75)", "R18+RMTL (1)",  "R50+RMTL (0.25)", "R50+RMTL (0.5)", "R50+RMTL (0.75)", "R50+RMTL (1)", "R50+LA-Net (0.25)", "R50+LA-Net (0.5)", "R50+LA-Net (0.75)", "R50+LA-Net (1)"]
        draw_nets = ["UNet", "R18+RMTL", "R50+RMTL", "R50+LA-Net"]
        # "R18+LA-Net (0.25)",, "R18+LA-Net (0.5)", "R18+LA-Net (0.75)", "R18+LA-Net (1)",
        targets = ["0.25", "0.5", "0.75", "1"]
        markers = ["x"] * len(draw_nets)
        # ax = [ax]

    color_palette = PART_DATA_PALETTE
    fig, ax = plt.subplots(1, len(metrics))
    fig.set_size_inches(6*len(metrics), 8) 
    df = pd.read_csv(csv_file, sep=",")
    df = df[df["Size"] == size]
    df[metrics] = df[metrics] * 100

    # draw bar
    for i, metric in enumerate(metrics):
        means, lows, highs, errs = [], [], [], []
        
        for j, net in enumerate(nets):
            m, err = get_confidence_interval(df, metric, net)
            means.append(m)
            lows.append(m-err)
            highs.append(m+err)
            errs.append(err)
        for k, net in enumerate(draw_nets):
            interval = len(nets) // len(draw_nets)
            xs = [str(s) for s in targets]
            # ax[i].plot(xs, means[interval*k:interval*(k+1)], "--{}".format(markers[k]), markersize=10, linewidth=2, label=net, color=color_palette[net]) 
            # draw confidence bands
            # ax[i].fill_between(xs, lows[interval*k:interval*(k+1)], highs[interval*k:interval*(k+1)], alpha=0.2, color=color_palette[net])
            ax[i].errorbar(xs, means[interval*k:interval*(k+1)], 
                           yerr=errs[interval*k:interval*(k+1)], 
                           fmt="--{}".format(markers[k]), markersize=10, linewidth=2, label=net, color=color_palette[net], capsize=7)
    # set subplot title
    for i, metric in enumerate(metrics):
        ax[i].set_title(metric)

    fig_idx = list(range(len(metrics)))
    for i in fig_idx:
        # if type == "part_data":
        #     ax[i].set_ylim([0.85, 1.01])
        #     ax[i].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
        # elif type == "part_data_iou":
        #     if size == 256:
        #         ax[i].set_ylim([0.36, 0.45])
        #         ax[i].yaxis.set_ticks(np.arange(0.36, 0.45, 0.02))
        #     elif size == 512:
        #         ax[i].set_ylim([0.45, 0.61])
        #         ax[i].yaxis.set_ticks(np.arange(0.45, 0.61, 0.05))
        ax[i].set_xticks(xs)
    handles, labels = ax[0].get_legend_handles_labels()
    if type == "part_data":
        # ax[0].legend(handles, labels, bbox_to_anchor=(0.5, -1.1, 5.5, 1), loc="upper left", ncol=len(draw_nets), mode="expand", borderaxespad=0, fontsize=32) 
        ax[1].legend(handles, labels, bbox_to_anchor=(0.2, -1.1, 3.2, 1), loc="upper left", ncol=len(draw_nets), mode="expand", borderaxespad=0, fontsize=34) 
    if type == "part_data_iou":
        ax[0].legend(handles, labels, bbox_to_anchor=(0.0, -1.1, 2.3, 1), loc="upper left", ncol=len(draw_nets), mode="expand", borderaxespad=0, fontsize=34) 
    plt.tight_layout()
    plt.savefig(image_name)


if __name__ == "__main__":
    # draw method 
    size = 256
    # img_name = "method_effect_mayo_{}_no_sig.png".format(size)
    # draw_method_effect(img_name, size=size, dataset="MAYO")

    # ablation
    # img_name = "method_effect_busi_{}_no_sig.png".format(size) 
    img_name = "ablation_mayo_size_boxplot.pdf"
    img_name = "test.png"
    draw_method_effect(img_name, size=size, dataset="mayo_bbox_cls")  

    # part data busi
    img_name = "ablation_busi_datasize_iou.pdf"
    # img_name = "test.png"
    # draw_confidence_band(img_name, size, type="part_data_iou")

    # pv = st.ttest_ind(a, b, alternative="less")
    # print(pv)
    # sample = np.array([0.563,	0.566,	0.5 ])
    # meanv = np.mean(sample)
    # std = st.sem(sample)
    # ci = st.t.interval(alpha=0.95, df=len(sample)-1, loc=meanv, scale=std)
    # print(meanv, (ci[1] - meanv)) # convert to %

    # a = [0.396,	0.409,	0.412]
    # b = [0.426,	0.426,	0.423]
    # pv = st.ttest_ind(a, b, alternative="less")
    # print(pv)
    