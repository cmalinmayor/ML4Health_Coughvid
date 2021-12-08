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
    o, curr_index = get_line_0(lines, curr_index)
    if o is None:
        return None, curr_index
    result.update(o)
    o, curr_index = get_line_1(lines, curr_index)
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
        
        
def get_line_0(lines, index):
    line_re = 'Report for epoch (\d*)'
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

    return {"epoch": epoch}, index

def get_line_1(lines, index):
    line_re = 'Train accuracy..(\d.\d*)'
    p = re.compile(line_re)
    match = None
    while(match is None and index < len(lines)):
        line = lines[index]
        match = p.match(line)
        index += 1
    if match is None:
        return None, index
    train_accuracy = match.group(1)

    return {"train accuracy": train_accuracy}, index

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

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Range1d


def plot_accuracy(df_da, df_baseline, title="Coswara Accuracy", max_epoch=100):
    source_da = ColumnDataSource(df_da)
    source_baseline = ColumnDataSource(df_baseline)
    p = figure(title=title, x_axis_label='Epoch', y_axis_label='Accuracy', plot_width=700, plot_height=700)

    p.line(x="epoch", y="train_acc", source=source_da, legend_label="augmentation train", line_width=2)
    p.line(x="epoch", y="test_balanced_acc", source=source_da, legend_label="augmentation test", line_width=2,
           color='Red')

    p.line(x="epoch", y="train_acc", source=source_baseline, legend_label="baseline train", line_width=2, color='Green')
    p.line(x="epoch", y="test_balanced_acc", source=source_baseline, legend_label="baseline test", line_width=2,
           color='Orange')

    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '14pt'
    p.legend.label_text_font_size = '10pt'
    p.title.text_font_size = '16pt'
    p.x_range = Range1d(0, max_epoch)
    p.legend.location = "top_left"
    show(p)

# %%
embedding_filenames = ['512_6_1e-5/stats.csv', ]#'1024_6_1e-5/stats.csv', '2048_6_5e-6/stats.csv']
embedding_labels= ['512', ] # '1024', '2048']

filename_baseline = 'baseline_stats.csv'

df_baseline = pd.read_csv(filename_baseline)
df_baseline

# %%
from bokeh.palettes import colorblind
print(colorblind)
def plot_embedding_accuracy(filenames, labels, baseline_df,
                            title="Accuracy of BYOL-A Pre-Trained Embeddings", max_epoch=100): 
    sources = [ColumnDataSource(pd.read_csv(f)) for f in filenames]
    source_baseline = ColumnDataSource(df_baseline)
    p = figure(title=title, x_axis_label='Epoch', y_axis_label='Accuracy', plot_width=700, plot_height=700)
    
    colors = colorblind["Colorblind"][7]
    
    p.line(x="epoch", y="train_acc", source=source_baseline, legend_label="baseline train",
           line_width=2, color=colors[0], line_dash='dashed')
    p.line(x="epoch", y="test_balanced_acc", source=source_baseline, legend_label="baseline test",
           line_width=2, color=colors[0])

    for index, pair in enumerate(zip(sources, labels)):
        source, name = pair
        p.line(x="epoch", y="train_acc", source=source, legend_label=f"{name} train",
               line_width=2, color=colors[index +1], line_dash='dashed')
        p.line(x="epoch", y="test_balanced_acc", source=source, legend_label=f"{name} test",
               line_width=2, color=colors[index+1], )
    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '14pt'
    p.legend.label_text_font_size = '10pt'
    p.title.text_font_size = '16pt'
    p.x_range = Range1d(0, max_epoch)
    p.legend.location = "top_left"
    show(p)
    


# %%
plot_embedding_accuracy(embedding_filenames, embedding_labels, df_baseline, title="Embedding Accuracy Comparison with Baseline")

# %%
df_baseline

# %%
df_aug = read_stats_from_file('../da')

# %%
f = '512_6_1e-5/90_prc.csv'
df = pd.read_csv(f)

# %%
df = df.drop(424)

# %%
pd.set_option("display.max_rows", None, "display.max_columns", None)
df


# %%
def plot_prcurve(df, title="Precision Recall Curve", max_epoch=100): 
    source_baseline = ColumnDataSource(df)
    p = figure(title=title, x_axis_label='Recall', y_axis_label='Precision', plot_width=700, plot_height=700)
    
    colors = ["#191970", "#006400", "#ff0000", "#ffd700"]
    
    p.line(x="Recall", y="Precision", source=source_baseline, legend_label="epochX",
           line_width=2, color=colors[0])

    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '14pt'
    p.legend.label_text_font_size = '10pt'
    p.title.text_font_size = '16pt'
    p.x_range = Range1d(0, 1)
    p.legend.location = "top_left"
    show(p)
    


# %%

plot_prcurve(df)

# %%
import sklearn.metrics
sklearn.metrics.auc(df['Recall'], df['Precision'])

# %%
stats_df = pd.read_csv('512_6_1e-5/stats.csv')
stats_df

# %%
stats_df[stats_df['test_balanced_acc'] == stats_df['test_balanced_acc'].max()]

# %%
