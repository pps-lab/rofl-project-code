#!/usr/bin/python
# coding=utf-8
import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

plot_data_save_path = "../data/plotsaves/"
plots = "../../images/"


def backup_array(arr, name):
    np.save(os.path.join(plot_data_save_path, name), arr)


def load_backup_array(name):
    return np.load(os.path.join(plot_data_save_path, name + ".npy"))


#############
# E2E PLOTS #
#############

PLAIN = "Plaintext"
PLAIN_S_CACHE = "Plaintext 1MB-Cache"
TIMECRYPT = "TimeCrypt"
TIMECRYPT_S_CACHE = "TimeCrypt 1MB-Cache"
GAMAL = "EC-ElGamal"
GAMAL_S_CACHE = "EC-ElGamal 1MB-Cache"
PAILLIER = "Paillier"
PAILLIER_S_CACHE = "Paillier 1MB-Cache"
TIMECRYPT_VERIFY = "TimeCrypt+"

e2e_raw_files_for_type = {
    PLAIN: "e2e_plain_berk_all.csv",
    PLAIN_S_CACHE: "e2e_small_cache_plain.csv",
    TIMECRYPT: "e2e_tc_berk_cenc.csv",
    TIMECRYPT_VERIFY: "e2e_tcmac_berk_v2.csv",
    TIMECRYPT_S_CACHE: "e2e_small_cache_castel.csv",
    GAMAL: "benchlog_ecelgamal.csv",
    PAILLIER: "benchlog_paillier_12x.csv",
    GAMAL_S_CACHE: "benchlog_ecelgamal_small_cache.csv",
    PAILLIER_S_CACHE: "benchlog_paillier_12x_small_cache.csv"

}

rate_for_type = {
    PLAIN: 100,
    TIMECRYPT: 100,
    TIMECRYPT_VERIFY: 100,
    GAMAL: 10,
    PAILLIER: 10
}



# DATA DUMP Functions

def e2e_summary_dump_data():
    data_path = "../data/end_to_end"

    fast_types = [PLAIN, PLAIN_S_CACHE, TIMECRYPT, TIMECRYPT_S_CACHE]
    slow_types = [GAMAL, GAMAL_S_CACHE, PAILLIER, PAILLIER_S_CACHE]

    line_pattern = re.compile("(.*),([0-9]+),(.*),([0-9]+)")
    skip = 1000

    def load_data(file_name):
        with open(os.path.join(data_path, file_name), 'rb') as file_reader:
            # key_numbers = np.asarray(map(lambda x: float(x), file_reader.readline().split(", ")))

            dataI = []
            dataQ = []
            count = -1
            fromTime = ""
            lastTs = ""
            for line in file_reader:
                m = line_pattern.match(line)
                if m:
                    count += 1
                    if count < skip:
                        continue
                    lastTs = m.group(1)
                    if count == skip:
                        fromTime = datetime.strptime(lastTs, "%H:%M:%S.%f")
                    if m.group(3) == "Q":
                        dataQ.append(float(m.group(4)))
                    else:
                        dataI.append(float(m.group(4)))

            toTime = datetime.strptime(lastTs, "%H:%M:%S.%f")
            dataI = np.asarray(dataI) / 1000000
            dataQ = np.asarray(dataQ) / 1000000
            return dataI, dataQ, (toTime - fromTime).total_seconds()

    for type in fast_types + slow_types:
        dataI, dataQ, time_frame = load_data(e2e_raw_files_for_type[type])
        backup_array(dataI, "e2e_%s_dataI" % (type,))
        backup_array(dataQ, "e2e_%s_dataQ" % (type,))
        backup_array(np.asarray([time_frame]), "e2e_%s_timeframe" % (type,))


def e2e_stat_dump():
    data_path = "../data/end_to_end"
    exp_name = "stat-query-mac-all.csv"

    def load_data(file_name):
        with open(os.path.join(data_path, file_name), 'rb') as file_reader:
            # key_numbers = np.asarray(map(lambda x: float(x), file_reader.readline().split(", ")))
            data = {}
            cur_group = 0
            cur_batch = []
            cur_type = None
            for line in file_reader:
                data_items = line.split(",")[1:]
                #data_items = map(lambda x: float(x), data_items[1:])


                if data_items[0] !=cur_group:
                    if data_items[2] == cur_type:
                        data[cur_type].append(cur_batch)
                    else:
                        if cur_type is not None:
                            data[cur_type].append(cur_batch)
                        cur_type = data_items[2]
                        data[cur_type] = []
                    cur_group = data_items[0]
                    cur_batch = [float(data_items[3])]
                else:
                    cur_batch.append(float(data_items[3]))
            for x in data:
                data[x] = np.asarray(data[x])[400:-400, :] / 1000000
            return data

    data = load_data(exp_name)
    dataP = data['plain']
    dataE = data['enc']
    dataEM = data['camac']
    backup_array(dataP, "e2e_stat_dataP")
    backup_array(dataE, "e2e_stat_dataE")
    backup_array(dataEM, "e2e_stat_dataEM")

def ml_plot_dump():
    types = ['plain', 'enc', 'encsb', 'camac']

    data_path = "../data/berkbench/ml_linreg_all_v2.csv"
    with open(data_path, 'rb') as file_reader:
        # key_numbers = np.asarray(map(lambda x: float(x), file_reader.readline().split(", ")))
        result = {}
        for x in types:
            result[x] = []
        cur_group = 0
        cur_batch = []
        x_axis = []
        for line in file_reader:
            data_items = line.split(",")
            data_items =data_items[1:]
            current_type = data_items[2]

            if int(data_items[0]) != cur_group:
                result[current_type].append(cur_batch)
                cur_group = int(data_items[0])
                cur_batch = [float(data_items[3])]
                x_axis = [float(data_items[1])]
            else:
                cur_batch.append(float(data_items[3]))
                x_axis.append(float(data_items[1]))

        for x in types:
            result[x] = np.asarray(result[x])[1000:9000, :] / 1000
            backup_array(np.median(result[x], axis=0), 'ml_plot_%s' % (x, ))
        backup_array(np.asarray(x_axis), 'ml_plot_xaxis')

# PLOTS

def e2e_latency_fast_plot(plot_name):
    def load_data(type):
        dataI = load_backup_array("e2e_%s_dataI" % (type,))
        dataQ = load_backup_array("e2e_%s_dataQ" % (type,))
        return dataI, dataQ

    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width * 0.67, fig_height / 1.22]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'font.size': 18,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}

    def set_box_color(bp, color, lw):
        plt.setp(bp['boxes'], color=color, linewidth=lw)
        plt.setp(bp['whiskers'], color=color, linewidth=lw)
        plt.setp(bp['caps'], color=color, linewidth=lw)
        plt.setp(bp['medians'], color=color, linewidth=lw)

    pdf_pages = PdfPages(os.path.join(plots, plot_name))
    fig_size = [fig_width / 1.8, fig_height / 1.6]

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    dataIP, dataQP = load_data(PLAIN)
    dataIPS, dataQPS = load_data(PLAIN_S_CACHE)
    dataIT, dataQT = load_data(TIMECRYPT)
    dataITS, dataQTS = load_data(TIMECRYPT_S_CACHE)

    width = 1.5

    colors = ['0.2', '0.5']

    bp1 = plt.boxplot([dataIP, dataIPS, dataQP, dataQPS], sym='', positions=np.array(range(4)) * 2.0 - 0.4,
                      widths=0.7)
    bp2 = plt.boxplot([dataIT, dataITS, dataQT, dataQTS], sym='', positions=np.array(range(4)) * 2.0 + 0.4,
                      widths=0.7)

    set_box_color(bp1, colors[0], width)
    set_box_color(bp2, colors[1], width)

    plt.plot([], c=colors[0], label='Plaintext', linewidth=2)
    plt.plot([], c=colors[1], label='TimeCrypt', linewidth=2)
    plt.legend(bbox_to_anchor=(-0.030, 0.94, 1., .102), loc=3, ncol=4, columnspacing=0.6, borderpad=0.3)

    plt.ylabel("Latency [ms]")
    plt.ylim(ymin=0, ymax=12)

    types = ["Insert", "Insert S", "Query", "Query S"]

    plt.xticks(range(0, len(types) * 2, 2), types)
    plt.xlim(-1, len(types) * 2 - 1)

    plt.gca().yaxis.grid(True, linestyle=':', color='0.8', zorder=0)
    #plt.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight', pad_inches=0)
    print("generated " + plot_name)
    plt.clf()
    pdf_pages.close()


def e2e_latency_slow_plot(plot_name):
    def load_data(type):
        dataI = load_backup_array("e2e_%s_dataI" % (type,))
        dataQ = load_backup_array("e2e_%s_dataQ" % (type,))
        return dataI, dataQ

    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width * 0.67, fig_height / 1.22]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'font.size': 18,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}

    def set_box_color(bp, color, lw):
        plt.setp(bp['boxes'], color=color, linewidth=lw)
        plt.setp(bp['whiskers'], color=color, linewidth=lw)
        plt.setp(bp['caps'], color=color, linewidth=lw)
        plt.setp(bp['medians'], color=color, linewidth=lw)

    pdf_pages = PdfPages(os.path.join(plots, plot_name))
    fig_size = [fig_width / 1.8, fig_height / 1.6]

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    dataIP, dataQP = load_data(PAILLIER)
    dataIPS, dataQPS = load_data(PAILLIER_S_CACHE)
    dataIT, dataQT = load_data(GAMAL)
    dataITS, dataQTS = load_data(GAMAL_S_CACHE)

    width = 1.5

    colors = ['0.2', '0.5']

    bp1 = plt.boxplot([dataIP, dataIPS, dataQP, dataQPS], sym='', positions=np.array(range(4)) * 2.0 - 0.4,
                      widths=0.7)
    bp2 = plt.boxplot([dataIT, dataITS, dataQT, dataQTS], sym='', positions=np.array(range(4)) * 2.0 + 0.4,
                      widths=0.7)

    set_box_color(bp1, colors[0], width)
    set_box_color(bp2, colors[1], width)

    plt.plot([], c=colors[0], label='Paillier', linewidth=2)
    plt.plot([], c=colors[1], label='EC-ElGamal', linewidth=2)
    plt.legend(bbox_to_anchor=(-0.030, 0.94, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.ylabel("Latency [ms]")
    plt.ylim(ymin=0, ymax=700)

    types = ["Insert", "Insert S", "Query", "Query S"]

    plt.xticks(range(0, len(types) * 2, 2), types)
    plt.xlim(-1, len(types) * 2 - 1)
    plt.gca().yaxis.grid(True, linestyle=':', color='0.8', zorder=0)
    #plt.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight', pad_inches=0)
    print("generated " + plot_name)
    plt.clf()
    pdf_pages.close()


def e2e_tp_insert(plot_name):
    types = [PLAIN, TIMECRYPT, TIMECRYPT_VERIFY, GAMAL, PAILLIER]

    def load_data(type):
        dataI = load_backup_array("e2e_%s_dataI" % (type,))
        dataQ = load_backup_array("e2e_%s_dataQ" % (type,))
        time_slot = load_backup_array("e2e_%s_timeframe" % (type,))
        return dataI, dataQ, time_slot

    def compute_tp(dataI, dataQ, time_slot, rate):
        tpI = (dataI.shape[0] * rate / time_slot) * 500
        tpQ = dataQ.shape[0] * rate / time_slot
        return tpI, tpQ

    tpi_for_type = []

    for type in types:
        dataI, dataQ, time_slot = load_data(type)
        tpI, _ = compute_tp(dataI, dataQ, time_slot, rate_for_type[type])
        tpi_for_type.append(tpI)

    # figure settings
    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width * 0.67, fig_height / 1.22]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 18,
              'xtick.labelsize': 12,
              'ytick.labelsize': 18,
              'font.size': 18,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}

    pdf_pages = PdfPages(os.path.join(plots, plot_name))
    fig_size = [fig_width / 2.0, fig_height / 1.6]

    plt.rcParams.update(params)
    # plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    f, ax1 = plt.subplots()
    # ax1.set_title('Insert Throughput')

    ind = np.arange(0, len(types)) * 1.2
    width = 0.8
    colours = ['0.1', '0.3', '0.5', '0.7', '0.9']


    # r1 = ax1.bar(ind - width / 2, np.asarray([tpI, tpQ]), width, color='0.6', alpha=0.8, align='edge')
    # r2 = ax1.bar(ind + width / 2, np.asarray([tpIP, tpQP]), width, color='0.2', alpha=0.8, align='edge')
    bars = plt.bar(ind, np.asarray(tpi_for_type), width, align='center', zorder=3)
    print("Insert Throughput")
    print(np.asarray(tpi_for_type).tolist())
    print("Insert Throughput overhead")
    print([x[0] / np.asarray(tpi_for_type)[0][0] for x in np.asarray(tpi_for_type).tolist()])

    for id, bar in enumerate(bars):
        bar.set_facecolor(colours[id])

    ax1.set_ylabel('Throughput [records/s]', position=(0, 0.4))
    ax1.set_xticks(ind)
    ax1.set_xticklabels(types, rotation=345)
    # ax1.set_xticklabels(('Insert', 'Query'))
    # ax1.legend(bars, types, loc=2)
    # ax1.set_ylim([0, 21000])
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.1fM' % (y * 1e-6)))
    ax1.yaxis.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)

    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight', pad_inches=0)
    print("generated " + plot_name)
    plt.clf()
    pdf_pages.close()


def e2e_tp_query(plot_name):
    types = [PLAIN, TIMECRYPT, TIMECRYPT_VERIFY, GAMAL, PAILLIER]

    def load_data(type):
        dataI = load_backup_array("e2e_%s_dataI" % (type,))
        dataQ = load_backup_array("e2e_%s_dataQ" % (type,))
        time_slot = load_backup_array("e2e_%s_timeframe" % (type,))
        return dataI, dataQ, time_slot

    def compute_tp(dataI, dataQ, time_slot, rate):
        tpI = (dataI.shape[0] * rate / time_slot) * 500
        tpQ = dataQ.shape[0] * rate / time_slot
        return tpI, tpQ

    tpq_for_type = []
    for type in types:
        dataI, dataQ, time_slot = load_data(type)
        _, tpQ = compute_tp(dataI, dataQ, time_slot, rate_for_type[type])
        tpq_for_type.append(tpQ)

    # figure settings
    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width * 0.67, fig_height / 1.22]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 18,
              'xtick.labelsize': 12,
              'ytick.labelsize': 18,
              'font.size': 18,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}

    pdf_pages = PdfPages(os.path.join(plots, plot_name))
    fig_size = [fig_width / 2.0, fig_height / 1.6]

    plt.rcParams.update(params)
    # plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    f, ax1 = plt.subplots()
    # ax1.set_title('Insert Throughput')

    ind = np.arange(0, len(types)) * 1.2
    width = 0.8
    colours = ['0.1', '0.3', '0.5', '0.7', '0.9']


    # r1 = ax1.bar(ind - width / 2, np.asarray([tpI, tpQ]), width, color='0.6', alpha=0.8, align='edge')
    # r2 = ax1.bar(ind + width / 2, np.asarray([tpIP, tpQP]), width, color='0.2', alpha=0.8, align='edge')
    bars = plt.bar(ind, np.asarray(tpq_for_type), width, align='center', zorder=3)
    print("Query Throughput")
    print(np.asarray(tpq_for_type).tolist())
    print("Query Throughput overhead")
    print([x[0] / np.asarray(tpq_for_type)[0][0] for x in np.asarray(tpq_for_type).tolist()])

    for id, bar in enumerate(bars):
        bar.set_facecolor(colours[id])

    ax1.set_ylabel('Throughput [ops/s]')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(types, rotation=345)
    # ax1.set_xticklabels(('Insert', 'Query'))
    # ax1.legend(bars, types, loc=2)
    # ax1.set_ylim([0, 21000])
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))
    ax1.yaxis.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)

    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight', pad_inches=0)
    print("generated " + plot_name)
    plt.clf()
    pdf_pages.close()


def e2e_statistical_queries(plot_name):
    types = ["minute", "hour", "day", "week", "month"]
    statistical_types_to_interval = {
        "min": 6,
        "hour": 360,
        "day": 8640,
        "week": 60480,
        "month": 241920
    }

    dataP = load_backup_array("e2e_stat_dataP")
    dataE = load_backup_array("e2e_stat_dataE")
    dataEM = load_backup_array("e2e_stat_dataEM")
    print(dataP)
    print(dataE)
    print(dataEM)

    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width, fig_height / 1.22]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 16,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'font.size': 18,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}

    pdf_pages = PdfPages(os.path.join(plots, plot_name))

    fig_size = [fig_width * 0.8, fig_height * 0.55]

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
        plt.setp(bp['fliers'], color=color)

    bp1 = plt.boxplot(dataP, 0, 'b+', positions=np.array(range(dataP.shape[1])) * 2.0-0.6, patch_artist=True,
                      widths=0.7)
    bp2 = plt.boxplot(dataE, 0, 'b+', positions=np.array(range(dataE.shape[1])) * 2.0, patch_artist=True,
                      widths=0.7)
    bp3 = plt.boxplot(dataEM, 0, 'b+', positions=np.array(range(dataEM.shape[1])) * 2.0 + 0.6, patch_artist=True,
                      widths=0.7)

    colours = ['0.1', '0.5', '0.7']

    set_box_color(bp1, colours[0])
    set_box_color(bp2, colours[1])
    set_box_color(bp3, colours[2])

    plt.plot([], c=colours[0], label='Plaintext', linewidth=2)
    plt.plot([], c=colours[1], label='TimeCrypt', linewidth=2)
    plt.plot([], c=colours[2], label='TimeCrypt+', linewidth=2)
    plt.legend(bbox_to_anchor=(-0.023, 0.93, 1., .102), loc=3, ncol=3, columnspacing=0.6, borderpad=0.3)

    plt.gca().yaxis.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)

    plt.yscale('log')
    plt.ylabel("Latency in [ms]")

    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))

    plt.xticks(range(0, len(types) * 2, 2), types)
    plt.xlim(-1, len(types) * 2 - 1)

    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight', pad_inches=0)
    print("generated " + plot_name)
    plt.clf()
    pdf_pages.close()


#######################
# KEYREGRESSION PLOTS #
#######################

SHA = "SHA256"
AES = "AES"
AESNI = "AES-NI"

keyreg_files = {
    SHA: "KeyReg_SHA256_i10000_k2.csv",
    AES: "KeyReg_AES_i10000_k2.csv",
    AESNI: "KeyReg_AESNi_i10000_k2.csv"
}


def keyreg_dump_data():
    data_path = "../data/key_regression/mac"
    types = [SHA, AES, AESNI]

    def load_data(type):
        file_name = keyreg_files[type]
        with open(os.path.join(data_path, file_name), 'rb') as file_reader:
            key_numbers = np.asarray([float(x) for x in file_reader.readline().split(", ")])
            data = []
            for line in file_reader:
                data.append([float(x) for x in line.split(", ")])
            data = np.asarray(data)
            data_avg = np.average(data, axis=0) / 1000
            data_err = np.std(data, axis=0) / 1000
            return np.asarray([key_numbers, data_avg, data_err])

    for type in types:
        key_reg_data = load_data(type)
        backup_array(key_reg_data, "keyreg_%s_data" % (type,))


def keyreg_latency_for_treesize(plot_name):
    types = [AES, SHA, AESNI]

    data_for_type = {}
    for type in types:
        data_for_type[type] = load_backup_array("keyreg_%s_data" % (type,))

    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width, fig_height / 1.22]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'font.size': 18,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}

    pdf_pages = PdfPages(os.path.join(plots, plot_name))
    fig_size = [fig_width, fig_height / 1.6]

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors = ['0.1', '0.6', '0.85']
    linestyles = ['-', '-', '-']
    dotstyle = ['o', 'v', '^']

    for id, type in enumerate(types):
        data = data_for_type[type]
        print(type)
        print("%.2f microsec for one key computation and %.2f microsec for enc/dec, %d" % (data[1, 30], 2 * data[1, 30], data[0, 30]))
        plt.loglog(data[0, 1:], data[1, 1:], dotstyle[id], label=type, color=colors[id], linestyle=linestyles[id], linewidth=2, zorder=3)

    plt.semilogx(1, 1, basex=2)
    plt.xlabel('Number of Keys')

    plt.ylabel("Time in [µs]")
    plt.ylim(ymin=0, ymax=100)
    plt.xlim(xmin=0)

    plt.legend(bbox_to_anchor=(-0.01, 0.82, 1., .102), loc=3, ncol=4, columnspacing=0.6, borderpad=0.3)

    plt.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight', pad_inches=0)
    print("generated " + plot_name)
    plt.clf()
    pdf_pages.close()


#####################
# MICRO INDEX PLOTS #
#####################

mirco_index_query_raw_files = {
    PLAIN: "micro_long_plain_all.csv",
    TIMECRYPT: "micro_long_cast_all.csv",
    GAMAL: "micro_ecelgamal_2_20.csv",
    PAILLIER: "micro_paillier_2_20.csv",
    TIMECRYPT_VERIFY: "micro_cmaca_all_26.csv"
}


def micro_index_dump_data():
    types = [PLAIN, TIMECRYPT, GAMAL, PAILLIER, TIMECRYPT_VERIFY]
    data_path_local = "../data/tree_micro_all_in_one"

    def load_data_all_in_one(type, data_path_local, files, data_for_phace=1, div=1000):
        def RepresentsInt(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        file_name = files[type]
        with open(os.path.join(data_path_local, file_name), 'rb') as file_reader:
            data = []
            cur_id = 0
            cur_batch = []
            powers = []
            round_before = 0
            phase = 0
            for line in file_reader:
                cur_line_data = line.split(",")[1:]
                cur_round = int(cur_line_data[0])
                if phase == 0:
                    if not RepresentsInt(cur_line_data[1]):
                        phase += 1
                else:
                    if cur_round < round_before:
                        phase += 1
                round_before = cur_round
                if phase == data_for_phace:
                    if int(cur_line_data[0]) != cur_id:
                        data.append(cur_batch)
                        cur_id = int(cur_line_data[0])
                        cur_batch = [float(cur_line_data[3])]
                    else:
                        cur_batch.append(float(cur_line_data[3]))
                    if cur_id == 0:
                        powers.append(float(cur_line_data[2]))
            skip = 1000
            data.append(cur_batch)
            data = np.asarray(data) / div
            print(data.shape)
            data_avg = np.average(data[0:1000], axis=0)
            data_err = np.std(data[0:1000], axis=0)
            return np.asarray([powers, data_avg, data_err])

    for type in types:
        key_reg_data = load_data_all_in_one(type, data_path_local, mirco_index_query_raw_files)
        backup_array(key_reg_data, "micro_index_%s_data" % (type,))


def micro_index_latency_for_interval(plot_name):
    types_short = [PAILLIER, GAMAL]
    types_long = [TIMECRYPT_VERIFY, TIMECRYPT, PLAIN]

    data_for_type = {}
    for type in types_short + types_long:
        data_for_type[type] = load_backup_array("micro_index_%s_data" % (type,))

    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width, fig_height / 1.22]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'font.size': 18,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}

    pdf_pages = PdfPages(os.path.join(plots, plot_name))
    fig_size = [fig_width, fig_height * 0.5]

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)

    colors = ['0.1', '0.3', '0.5', '0.7', '0.8']
    dostyle = ['o', 'x', 'v', '^', '*']
    linestyles = ['--', '-', '--', '-', '--']
    lines = []

    for id, type in enumerate(types_long):
        data = data_for_type[type]
        curP, = ax1.plot(data[0, :], data[1, :], dostyle[id], label=type, color=colors[id], linestyle=linestyles[id],
                         linewidth=2, zorder=3)
        lines.append(curP)
        ax1.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)
        ax1.set_ylabel("[µs]")
        ax1.semilogx(1, 1, basex=2)
        ax1.set_ylim(0, 25)

    for id, type in enumerate(types_short):
        data = data_for_type[type]
        curP, = ax2.plot(data[0, :], data[1, :] / 1000, dostyle[id+ len(types_long)], label=type, color=colors[id + len(types_long)],
                         linestyle=linestyles[id + len(types_long)], linewidth=2, zorder=3)
        lines.append(curP)
        ax2.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)
        ax2.set_ylabel("[ms]")
        ax2.semilogx(1, 1, basex=2)
        ax2.set_ylim(0, 20)
        #ax2.set_xlim(1, 1 << 20)

    plt.subplots_adjust(hspace=0.12)
    # plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xlabel('Interval size')

    # plt.ylim(ymin=0, ymax=20)
    f.legend(lines, types_long + types_short, bbox_to_anchor=(0.01, 1.05 , 1., .102), loc=3, ncol=3, columnspacing=0.6,
             borderpad=0.2, handlelength=1.85)

    plt.setp(ax1.get_yticklabels()[-1], visible=False)
    plt.setp(ax2.get_yticklabels()[-1], visible=False)

    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight', pad_inches=0)
    print("generated " + plot_name)
    plt.clf()
    pdf_pages.close()
    

def ml_query_plot(plotname):
    x_axis = load_backup_array('ml_plot_xaxis')
    x_axis = x_axis * 500
    plot_types = ['plain', 'enc', 'camac']
    plot_legend = {'plain': 'Plaintext', 'enc': 'TimeCrypt', 'camac': 'TimeCrypt+'}

    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width, fig_height / 1.22]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'font.size': 18,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}

    pdf_pages = PdfPages('../plots/%s' % plotname)
    fig_size = [fig_width, fig_height / 1.9]

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors = ['0.1', '0.3', '0.6']
    linestyles = ['-', '--', '-']
    f, ax1 = plt.subplots()

    for id, type in enumerate(plot_types):
        data = load_backup_array('ml_plot_%s' % type)
        data = 1000000 / data
        plt.semilogx(x_axis, data, '-o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)

    plt.xlabel('Number of Records per Range')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Queries [ops/s]")
    plt.ylim(ymin=8000, ymax=12000)
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end + 1, 1000))
    plt.xlim(xmin=0, xmax=40000000)

    plt.legend(bbox_to_anchor=(-0.016, 0.72, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


if __name__ == "__main__":
    # e2e Plots
    selection = 'all'
    if len(sys.argv) > 1:
        selection = sys.argv[1]

    if selection == 'e2e_latency_fast_schemes' or selection == 'all':
        e2e_latency_fast_plot("paper_e2e_latency_fast_schemes.pdf")

    if selection == 'e2e_latency_slow_schemes' or selection == 'all':
        e2e_latency_slow_plot("paper_e2e_latency_slow_schemes.pdf")

    if selection == 'e2e_tp_insert' or selection == 'all':
        e2e_tp_insert("paper_e2e_tp_insert.pdf")

    if selection == 'e2e_tp_query' or selection == 'all':
        e2e_tp_query("paper_e2e_tp_query.pdf")

    if selection == 'e2e_statistical_query' or selection == 'all':
        e2e_statistical_queries("paper_e2e_statistical_query.pdf")

    if selection == 'keyreg_latency_treesize' or selection == 'all':
        keyreg_latency_for_treesize("paper_keyreg_latency_treesize.pdf")

    if selection == 'micro_index_query_latency' or selection == 'all':
        micro_index_latency_for_interval("paper_micro_index_query_latency.pdf")

    if selection == 'ml_plot' or selection == 'all':
        ml_query_plot("ml_plot_latency.pdf")
