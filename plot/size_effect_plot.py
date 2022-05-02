import matplotlib.pyplot as plt 
import pandas as pd 
import numpy  as np
import itertools 

plt.rcParams.update({'font.size': 28})
plt.rc('legend', fontsize=24)
plt.rc('axes', labelsize=26)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=26)    # fontsize of the tick labels
plt.rc('ytick', labelsize=26)


def flip_legned(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def draw_size_depth(image_name):
    csv_file = "size_effect.csv"
    df = pd.read_csv(csv_file, sep=" ")
    net = df.Network
    input_size = df.InputSize
    prec = df.Precision / 100
    sens = df.Sensitivity / 100
    spec = df.Specificity / 100
    f1 = df["F1-Score"] /100
    prec_error = df["Acc-Error"] /100
    sens_error = df["Sens-Error"]/100
    speci_error = df["Speci-Error"] /100
    f1_error = df['F1-Error']/100
    # set 3 groups 
    x = np.arange(len(net) // 4)
    x_labels = ["Size: {}".format(int(x)) for x in input_size[0::4]]
    bar_width = 0.15 
    sep = 0.04

    def _write_value(rects, ax):
        for rect in rects:
            h = rect.get_height() 
            ax.annotate("{}".format(int(h)), 
                        xy=(rect.get_x()+rect.get_width()/2, h),
                        xytext=(0.5, 1), 
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize="medium",
                        rotation=75
            )
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(14, 10)
    # rects_prec, rects_sens = [], []
    
    net_color = ["teal", "darkslategray", "lightcoral", "brown"]
    net_color = ["lightskyblue", "deepskyblue", "coral", "orangered"]
    net_color = ["#DBF2D5", "#9ED386", "#FFE4D4", "#FFAC7A"]
    for i in range(4):
        net_prec = ax[0,0].bar(x+bar_width*(i-1.5), prec[i::4], width=bar_width, label=net[i], color=net_color[i], yerr=prec_error[i::4])
        net_sens = ax[0,1].bar(x+bar_width*(i-1.5), sens[i::4], width=bar_width, label=net[i], color=net_color[i], yerr=sens_error[i::4])
        net_spec = ax[1,0].bar(x+bar_width*(i-1.5), spec[i::4], width=bar_width, label=net[i], color=net_color[i], yerr=speci_error[i::4])
        net_f1 = ax[1,1].bar(x+bar_width*(i-1.5), f1[i::4], width=bar_width, label=net[i], color=net_color[i], yerr=f1_error[i::4])
        # _write_value(net_prec, ax[0,0])
        # _write_value(net_sens, ax[0,1])
        # _write_value(net_spec, ax[1,0])
        # _write_value(net_f1, ax[1,1])

    # ax[0,0].set_ylabel("Precision (%)")
    # ax[0,1].set_ylabel("Sensitivity (%)")
    # ax[1,0].set_ylabel("Specificity (%)")
    # ax[1,1].set_ylabel("F1-Score (%)")
    ax[0,0].set_title("Precision")
    ax[0,1].set_title("Sensitivity")
    ax[1,0].set_title("Specificity")
    ax[1,1].set_title("F1-Score")
    fig_idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i,j in fig_idx:
        ax[i,j].set_ylim([0.85, 1.01])
        ax[i,j].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
        ax[i,j].set_xticks(x)
        ax[i,j].set_xticklabels(x_labels)
    ax[1, 0].legend(bbox_to_anchor=(0.15, -0.7, 2, 0.5), loc="upper left", ncol=2, mode="expand", borderaxespad=0)
    fig.tight_layout()
    # plt.show()
    plt.savefig(image_name)

def draw_net_effect(image_name):
    csv_file = "net_effect_on_covid_class_448.csv"

    df = pd.read_csv(csv_file, sep=" ")
    net = df.Network
    prec = df.Precision / 100
    sens = df.Sensitivity / 100
    spec = df.Specificity / 100
    f1 = df["F1-Score"] /100
    prec_error = df["Acc-Error"] /100
    sens_error = df["Sens-Error"]/100
    speci_error = df["Speci-Error"] /100
    f1_error = df['F1-Error']/100
    # set 3 groups 
    x = np.arange(len(net) // 4)
    # x_labels = ["Size: {}".format(x) for x in input_size[0::4]]
    bar_width = 0.15 
    sep = 0.04
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(16, 6)
    # rects_prec, rects_sens = [], []
    
    net_color = ["teal", "darkslategray", "lightcoral", "brown"]
    net_color = ["lightskyblue", "steelblue", "deepskyblue", "coral", "darksalmon","orangered"]
    net_color = ["#DBF2D5", "#BEE2AF", "#9ED386", "#FFE4D4", "#FFC9A8", "#FFAC7A"]
    # net_color = ["#DBF2D5", "#9ED386", "#FFE4D4", "#FFAC7A"]  
    for i in range(6):
        if i>= 3:
            c = i+2
        else:
            c = i
        net_prec = ax[0].bar(x+bar_width*(c-1.5), prec[i], width=bar_width, label=net[i], color=net_color[i], yerr=prec_error[i])
        net_sens = ax[1].bar(x+bar_width*(c-1.5), sens[i], width=bar_width, label=net[i], color=net_color[i], yerr=sens_error[i])
        net_spec = ax[2].bar(x+bar_width*(c-1.5), spec[i], width=bar_width, label=net[i], color=net_color[i], yerr=speci_error[i])
        net_f1 = ax[3].bar(x+bar_width*(c-1.5), f1[i], width=bar_width, label=net[i], color=net_color[i], yerr=f1_error[i])

    # ax[0].set_ylabel("Precision (%)")
    # ax[1].set_ylabel("Sensitivity (%)")
    # ax[2].set_ylabel("Specificity (%)")
    # ax[3].set_ylabel("F1-Score (%)")
    ax[0].set_title("Precision")
    ax[1].set_title("Sensitivity")
    ax[2].set_title("Specificity")
    ax[3].set_title("F1-Score")
    # fig_idx = [(0, 0), (0, 1), (0, 2), (0, 3)]
    for i in range(4):
        ax[i].set_ylim([0.85, 1.01])
        ax[i].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
        ax[i].set_xticks([])
    ax[0].legend(bbox_to_anchor=(0.5, -2.1, 4, 2), loc="upper left", ncol=2, mode="expand", borderaxespad=0)
    fig.tight_layout()
    # plt.show()
    plt.savefig(image_name)

def draw_method_effect(image_name, group_net=True):
    # csv_file = "diff_method_effect_224.csv"
    csv_file = "net_depth_224.csv"
    # csv_file = "net_depth_224_tfs.csv"
    df = pd.read_csv(csv_file, sep="\s+")
    net = df.Network
    num_net = len(net)
    group_interval = num_net // 2 # set groups
    prec = df.Precision / 100
    sens = df.Sensitivity / 100
    spec = df.Specificity / 100
    f1 = df["F1-Score"] /100
    prec_error = df["Acc-Error"] /100
    sens_error = df["Sens-Error"]/100
    speci_error = df["Speci-Error"] /100
    f1_error = df['F1-Error']/100
    # add auc
    # auc = df["AUC"] / 100
    # auc_error = df["AUC-Error"] / 100 
    # set N groups according to the number of nets
    if group_net:
        x = np.arange(len(net) // num_net)
    else:
        x = np.arange(len(net) // num_net)
    # x_labels = ["Size: {}".format(int(x)) for x in input_size[0::4]]
    bar_width = 0.15 
    sep = 0.04

    def _write_value(rects, ax):
        for rect in rects:
            h = rect.get_height() 
            ax.annotate("{}".format(int(h)), 
                        xy=(rect.get_x()+rect.get_width()/2, h),
                        xytext=(0.5, 1), 
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize="medium",
                        rotation=75
            )
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 16)
    # fig, ax = plt.subplots(1, 5)
    # fig.set_size_inches(35, 8)
    # rects_prec, rects_sens = [], []
    
    net_color = ["teal", "darkslategray", "lightcoral", "brown"]
    net_color = ["lightskyblue", "deepskyblue", "coral", "orangered"]
    net_color = ["#EDEDED", "#DBEBF8", "#FFF2C7", "#D8E3F5", "#FFD8EE", "#AFFFE8"]
    # for method
    net_color = ["#DFF1D7", "#5BB033", "#448427", "#2D581A", "#BEE2AF", "#9ED386", "#FFE4D4", "#FD7603", "#D45201", "#8E3701", "#FFC9A8", "#FFAE84"]
    net_color = ["#DFF1D7", "#5BB033", "#448427", "#2D581A", "#9ED386", "#FFE4D4", "#FD7603", "#D45201", "#8E3701", "#FFAE84"] 
    # for depth
    net_color = ["#D0CECE", "#FFE4D4", "#DBEBF8", "#FFF2C7", "#DFF1D7", "#777171", "#FFAE84", "#94C4EA", "#FFD84C", "#9ED386"]
    # # for tfs vs tl
    net_color = ["#FFE4D4", "#FFD8A0", "#DBEBF8", "#BBE9FE", "#DFF1D7", "#B7FFE6", "#FFAE84", "#FFA101", "#94C4EA", "#37B3FC","#9ED386", "#01FDB8"]

    if group_net:
        for i in range(num_net):
            if i>= group_interval:
                c = i+2
            else:
                c = i
            net_prec = ax[0,0].bar(x+bar_width*(c-1.5), prec[i], width=bar_width, label=net[i], color=net_color[i], yerr=prec_error[i])
            net_sens = ax[0,1].bar(x+bar_width*(c-1.5), sens[i], width=bar_width, label=net[i], color=net_color[i], yerr=sens_error[i])
            net_spec = ax[1,0].bar(x+bar_width*(c-1.5), spec[i], width=bar_width, label=net[i], color=net_color[i], yerr=speci_error[i])
            net_f1 = ax[1,1].bar(x+bar_width*(c-1.5), f1[i], width=bar_width, label=net[i], color=net_color[i], yerr=f1_error[i])
            # net_prec = ax[0].bar(x+bar_width*(c-1.5), prec[i], width=bar_width, label=net[i], color=net_color[i], yerr=prec_error[i])
            # net_sens = ax[1].bar(x+bar_width*(c-1.5), sens[i], width=bar_width, label=net[i], color=net_color[i], yerr=sens_error[i])
            # net_spec = ax[2].bar(x+bar_width*(c-1.5), spec[i], width=bar_width, label=net[i], color=net_color[i], yerr=speci_error[i])
            # net_f1 = ax[3].bar(x+bar_width*(c-1.5), f1[i], width=bar_width, label=net[i], color=net_color[i], yerr=f1_error[i])
            # net_auc = ax[4].bar(x+bar_width*(c-1.5), auc[i], width=bar_width, label=net[i], color=net_color[i], yerr=auc_error[i])
            # _write_value(net_prec, ax[0,0])
            # _write_value(net_sens, ax[0,1])
            # _write_value(net_spec, ax[1,0])
            # _write_value(net_f1, ax[1,1])
    else:
        for i in range(group_interval):
            c = i + 2*i
            # prec
            ax[0,0].bar(x+bar_width*(c-1.5), prec[i], width=bar_width, label=net[i], color=net_color[i], yerr=prec_error[i])
            ax[0,0].bar(x+bar_width*(c+1-1.5), prec[i+group_interval], width=bar_width, label=net[i+group_interval], color=net_color[i+group_interval], yerr=prec_error[i+group_interval])
            # sens
            ax[0,1].bar(x+bar_width*(c-1.5), sens[i], width=bar_width, label=net[i], color=net_color[i], yerr=sens_error[i])
            ax[0,1].bar(x+bar_width*(c+1-1.5), sens[i+group_interval], width=bar_width, label=net[i+group_interval], color=net_color[i+group_interval], yerr=sens_error[i+group_interval])
            # spec
            ax[1,0].bar(x+bar_width*(c-1.5), spec[i], width=bar_width, label=net[i], color=net_color[i], yerr=speci_error[i])
            ax[1,0].bar(x+bar_width*(c+1-1.5), spec[i+group_interval], width=bar_width, label=net[i+group_interval], color=net_color[i+group_interval], yerr=speci_error[i+group_interval])
            # f1
            ax[1,1].bar(x+bar_width*(c-1.5), f1[i], width=bar_width, label=net[i], color=net_color[i], yerr=f1_error[i])
            ax[1,1].bar(x+bar_width*(c+1-1.5), f1[i+group_interval], width=bar_width, label=net[i+group_interval], color=net_color[i+group_interval], yerr=f1_error[i+group_interval])

    # ax[0,0].set_ylabel("Precision (%)")
    # ax[0,1].set_ylabel("Sensitivity (%)")
    # ax[1,0].set_ylabel("Specificity (%)")
    # ax[1,1].set_ylabel("F1-Score (%)")
    ax[0,0].set_title("Precision")
    ax[0,1].set_title("Sensitivity")
    ax[1,0].set_title("Specificity")
    ax[1,1].set_title("F1-Score")
    fig_idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i,j in fig_idx:
        ax[i,j].set_ylim([0.85, 1.01])
        ax[i,j].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
        ax[i,j].set_xticks([])
        # ax[i,j].set_xticklabels(x_labels)
    handles, labels = ax[1, 0].get_legend_handles_labels()
    if not group_net:
        handles = flip_legned(handles, 2)
        labels = flip_legned(labels, 2)
    # ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.35, -0.7, 1.6, 0.5), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=20)
    ax[1, 0].legend(handles, labels, 
    bbox_to_anchor=(0.6, -1.1, 1.1, 1), 
    loc="upper left", ncol=1, mode="expand", borderaxespad=0, fontsize=20) 

    # ax[0].set_title("Precision")
    # ax[1].set_title("Sensitivity")
    # ax[2].set_title("Specificity")
    # ax[3].set_title("F1-Score")
    # ax[4].set_title("AUC")
    # fig_idx = list(range(5))

    # for i in fig_idx:
    #     ax[i].set_ylim([0.85, 1.01])
    #     ax[i].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
    #     ax[i].set_xticks([])
    #     # ax[i,j].set_xticklabels(x_labels)
    # handles, labels = ax[2].get_legend_handles_labels()
    # if not group_net:
    #     handles = flip_legned(handles, 2)
    #     labels = flip_legned(labels, 2)
    # ax[0].legend(handles, labels, bbox_to_anchor=(-0.05, -1.1, 6, 1), loc="upper left", ncol=10, mode="expand", borderaxespad=0, fontsize=20)

    fig.tight_layout()
    # plt.show()
    plt.savefig(image_name)


def draw_data_size_effect(image_name):
    csv_file = "data_size_224.csv"
    df = pd.read_csv(csv_file, sep="\s+")
    nets = np.unique(df.Network)
    markers = ["o", "v"]
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(14, 10)
    colors = ["blue", "green"]
    for i, net in enumerate(nets):
        size = df.Size[df.Network == net]
        prec = df.Precision[df.Network==net] / 100
        sens = df.Sensitivity[df.Network==net] / 100
        spec = df.Specificity[df.Network==net] / 100
        f1   = df["F1-Score"][df.Network==net] /100
        prec_error  = df["Acc-Error"][df.Network==net]/100
        sens_error  = df["Sens-Error"][df.Network==net]/100
        speci_error = df["Speci-Error"][df.Network==net]/100
        f1_error    = df['F1-Error'][df.Network==net]/100

        net_prec = ax[0,0].plot(size, prec, "-{}".format(markers[i]), markersize=10, linewidth=2, label=net, color=colors[i])
        ax[0,0].errorbar(size, prec, yerr=prec_error, fmt="none", linewidth=2)
        net_sens = ax[0,1].plot(size, sens, "-{}".format(markers[i]), markersize=10, linewidth=2, label=net, color=colors[i])
        ax[0,1].errorbar(size, sens, yerr=sens_error, fmt="none", linewidth=2, color=colors[i])
        net_spec = ax[1,0].plot(size, spec, "-{}".format(markers[i]), markersize=10, linewidth=2, label=net, color=colors[i])
        ax[1,0].errorbar(size, spec, yerr=speci_error, fmt="none",linewidth=2, color=colors[i])
        net_f1   = ax[1,1].plot(size,   f1, "-{}".format(markers[i]), markersize=10, linewidth=2, label=net, color=colors[i])
        ax[1,1].errorbar(size, f1, yerr=f1_error, fmt="none", linewidth=2)
       
    ax[0,0].set_ylabel("Precision")
    ax[0,1].set_ylabel("Sensitivity")
    ax[1,0].set_ylabel("Specificity")
    ax[1,1].set_ylabel("F1-Score")
    # ax[0,0].set_xlabel("Data size ratio")
    # ax[0,1].set_xlabel("Data size ratio")
    # ax[1,0].set_xlabel("Data size ratio")
    # ax[1,1].set_xlabel("Data size ratio")
    fig_idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i,j in fig_idx:
        ax[i,j].set_ylim([0.85, 1.01])
        ax[i,j].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
        ax[i,j].set_xlim([0.2, 1.05])
        ax[i,j].xaxis.set_ticks(np.arange(0.25, 1.01, 0.25))
        # ax[i,j].set_xticklabels(size)

    handles, labels = ax[1, 0].get_legend_handles_labels()
    ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.35, -0.7, 1.6, 0.5), loc="upper left", ncol=2, mode="expand", borderaxespad=0, fontsize=20)
    fig.tight_layout()
    # plt.show()
    plt.savefig(image_name)


def draw_HN_method_effect(image_name):
    csv_file = "diff_method_effect_32_HN.csv"
    df = pd.read_csv(csv_file, sep="\s+")
    net = df.Network
    num_net = len(net)
    prec = df.Precision / 100
    sens = df.Sensitivity / 100
    spec = df.Specificity / 100
    f1 = df["F1-Score"] /100
    prec_error = df["Acc-Error"] /100
    sens_error = df["Sens-Error"]/100
    speci_error = df["Speci-Error"] /100
    f1_error = df['F1-Error']/100
    # add auc
    auc = df["AUC"] / 100
    auc_error = df["AUC-Error"] / 100 
    # set N groups according to the number of nets
    x = np.arange(len(net))
    bar_width = 0.15 
    sep = 0.04

    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(20, 4)
    
    # for method
    net_color = ["#DFF1D7", "#5BB033", "#448427", "#2D581A", "#BEE2AF", "#9ED386", "#FFE4D4", "#FD7603", "#D45201", "#8E3701", "#FFC9A8", "#FFAE84"]
    net_color = ["#DFF1D7", "#5BB033", "#448427", "#2D581A", "#9ED386", "#FFE4D4", "#FD7603", "#D45201", "#8E3701", "#FFAE84"] 

    for i in range(num_net):
        net_prec = ax[0].bar(bar_width*(i)+sep*i, prec[i], width=bar_width, label=net[i], color=net_color[i], yerr=prec_error[i])
        net_sens = ax[1].bar(bar_width*(i)+sep*i, sens[i], width=bar_width, label=net[i], color=net_color[i], yerr=sens_error[i])
        net_spec = ax[2].bar(bar_width*(i)+sep*i, spec[i], width=bar_width, label=net[i], color=net_color[i], yerr=speci_error[i])
        net_f1 = ax[3].bar(bar_width*(i)+sep*i, f1[i], width=bar_width, label=net[i], color=net_color[i], yerr=f1_error[i])
        net_auc = ax[4].bar(bar_width*(i)+sep*i, auc[i], width=bar_width, label=net[i], color=net_color[i], yerr=auc_error[i])

    ax[0].set_title("Precision")
    ax[1].set_title("Sensitivity")
    ax[2].set_title("Specificity")
    ax[3].set_title("F1-Score")
    ax[4].set_title("AUC")
    fig_idx = list(range(5))

    for i in fig_idx:
        ax[i].set_ylim([0.85, 1.01])
        ax[i].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
        ax[i].set_xticks([])
        # ax[i,j].set_xticklabels(x_labels)
    handles, labels = ax[2].get_legend_handles_labels()
    ax[0].legend(handles, labels, bbox_to_anchor=(0.5, -1.1, 6, 1), loc="upper left", ncol=10, mode="expand", borderaxespad=0, fontsize=20)

    fig.tight_layout()
    # plt.show()
    plt.savefig(image_name)


def draw_net_effect_binary(image_name):
    csv_file = "net_depth_512_busi.csv"
    
    df = pd.read_csv(csv_file, sep=" ")
    net = df.Network
    prec = df.Precision / 100
    sens = df.Sensitivity / 100
    spec = df.Specificity / 100
    f1 = df["F1-Score"] /100
    prec_error = df["Acc-Error"] /100
    sens_error = df["Sens-Error"]/100
    speci_error = df["Speci-Error"] /100
    f1_error = df['F1-Error']/100
    bar_width = 0.15 
    sep = 0.04
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(16, 6)
    # rects_prec, rects_sens = [], []
    
    net_color = ["#DBF2D5", "#9ED386", "#FFE4D4", "#FFAC7A"]
    
    for i in range(4):
        net_prec = ax[0].bar(bar_width*(i)+i*sep, prec[i], width=bar_width, label=net[i], color=net_color[i], yerr=prec_error[i])
        net_sens = ax[1].bar(bar_width*(i)+i*sep, sens[i], width=bar_width, label=net[i], color=net_color[i], yerr=sens_error[i])
        net_spec = ax[2].bar(bar_width*(i)+i*sep, spec[i], width=bar_width, label=net[i], color=net_color[i], yerr=speci_error[i])
        net_f1 = ax[3].bar(bar_width*(i)+i*sep, f1[i], width=bar_width, label=net[i], color=net_color[i], yerr=f1_error[i])

    ax[0].set_title("Precision")
    ax[1].set_title("Sensitivity")
    ax[2].set_title("Specificity")
    ax[3].set_title("F1-Score")
    # fig_idx = [(0, 0), (0, 1), (0, 2), (0, 3)]
    for i in range(4):
        ax[i].set_ylim([0.80, 1.01])
        ax[i].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
        ax[i].set_xticks([])
    ax[0].legend(bbox_to_anchor=(0, -2.1, 5.5, 2), loc="upper left", ncol=4, mode="expand", borderaxespad=0)
    fig.tight_layout()
    # plt.show()
    plt.savefig(image_name)

if __name__ == "__main__":
    # img_name = "size_depth_effect_ssim.png"
    # # draw_size_depth(img_name)
    # img_name = "net_effect_ssim_448.png"
    # draw_net_effect(img_name)
    # img_name = "net_depth_224_2.png"
    # draw_method_effect(img_name, group_net=True)
    # img_name = "data_size_224.png"
    # draw_data_size_effect(img_name)
    # img_name = "method_effect_32_HN.png"
    # draw_HN_method_effect(img_name)
    img_name = "net_effect_512_busi.png"
    draw_net_effect_binary(img_name)