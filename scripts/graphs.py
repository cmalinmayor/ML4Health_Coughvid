# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import csv
import re


# %%
def read_stats_from_log(filename):
    results = []
    with open(filename) as f:
        lines = f.readlines()
        curr_index = 0
        result = {}
        while result is not None:
            result, curr_index = get_stats_for_epoch(lines, curr_index)
            if result is not None:
                results.append(result)
    return pd.DataFrame.from_dict(results)

def get_stats_for_epoch(lines, start_index):
    result = {}
    curr_index = start_index
    o, curr_index = get_line_1(lines, curr_index)
    if o is None:
        return None, curr_index
    result.update(o)
    o, curr_index = get_line_2(lines, curr_index)
    result.update(o)
    o, curr_index = get_line_3(lines, curr_index)
    result.update(o)
    o, curr_index = get_line_4(lines, curr_index)
    result.update(o)
    o, curr_index = get_line_5(lines, curr_index)
    result.update(o)
    o, curr_index = get_line_6(lines, curr_index)
    result.update(o)
    return result, curr_index
        
        
def get_line_1(lines, index):
    line_re = 'epoch: (\d*)...train loss..(\d.\d*)..train acc..(\d.\d*)'
    p = re.compile(line_re)
    match = None
    while(match is None and index < len(lines)):
        line = lines[index]
        match = p.match(line)
        index += 1
    if match is None:
        return None, index
    # print(f"Found match at index {index}")
    epoch = match.group(1)
    train_loss = float(match.group(2))
    train_acc = float(match.group(3))
    return {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}, index

def get_line_2(lines, index):   
    line_re = "Accuracy at threshold (\d.\d*)..(\d.\d*)"
    p = re.compile(line_re)
    match = None
    while(match is None):
        line = lines[index]
        match = p.match(line)
        index += 1
    if match is None:
        print(f'regex {line_re} did not match {line}')
        raise ValueError()
    threshold = float(match.group(1))
    test_acc = float(match.group(2))
    return {"threshold": threshold, "test_acc": test_acc}, index

def get_line_3(lines, index):   
    line_re = "Balanced a?ccuracy at threshold (\d.\d*)..(\d.\d*)"
    p = re.compile(line_re)
    match = None
    while(match is None):
        line = lines[index]
        match = p.match(line)
        index += 1
    if match is None:
        print(f'regex {line_re} did not match {line}')
        raise ValueError()
    threshold = float(match.group(1))
    test_acc = float(match.group(2))
    return {"threshold": threshold, "balanced_test_acc": test_acc}, index

def get_line_4(lines, index):   
    line_re = "Confusion matrix at threshold (\d\.\d*): \[\[\s*(\d+)\s+(\d+)\]"
    p = re.compile(line_re)
    match = None
    while(match is None):
        line = lines[index]
        match = p.match(line)
        index += 1
    if match is None:
        print(f'regex {line_re} did not match {line}')
        raise ValueError()
    tn = int(match.group(2))
    fp = int(match.group(3))
    return {"tn": tn, "fp": fp}, index

def get_line_5(lines, index):   
    line_re = ".\[\s*(\d+)\s+(\d+)\]"
    p = re.compile(line_re)
    match = None
    while(match is None):
        line = lines[index]
        match = p.match(line)
        index += 1
    if match is None:
        print(f'regex {line_re} did not match {line}')
        raise ValueError()
    fn = int(match.group(1))
    tp = int(match.group(2))
    return {"fn": fn, "tp": tp}, index

def get_line_6(lines, index):
    line_re = "AUC-ROC: (\d.\d+$)"
    p = re.compile(line_re)
    match = None
    while(match is None):
        line = lines[index]
        match = p.match(line)
        index += 1
    if match is None:
        # print(f'regex {line_re} did not match {line}')
        raise ValueError()
    auc_roc = float(match.group(1))
    return {"auc_roc": auc_roc}, index


# %%

filename_augmentation = '../da_log1_augmentation_result.txt'
filename_basic = '../train_100_epochs_filtered.log'

basic_df = read_stats_from_log(filename_basic)
aug_df = read_stats_from_log(filename_augmentation)


# %%
basic_df

# %%
aug_df


# %%
def read_stats_from_csv(filename):
    return pd.read_csv(filename)


# %%
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Range1d

def plot_accuracy(df, title="Coswara Accuracy", max_epoch=100):
    source = ColumnDataSource(df)
    p = figure(title=title, x_axis_label='Epoch', y_axis_label='Accuracy')

    p.line(x="epoch", y="train_acc", source=source, legend_label="train", line_width=2)
    p.line(x="epoch", y="balanced_test_acc", source=source, legend_label="test", line_width=2, color="red")
    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '14pt'
    p.legend.label_text_font_size = '14pt'
    p.title.text_font_size = '16pt'
    p.x_range = Range1d(0, max_epoch)
    p.legend.location = "top_left"
    show(p)


# %%
plot_accuracy(basic_df, title="Baseline Accuracy During Training")

# %%
