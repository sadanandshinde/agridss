from bokeh.plotting import figure, show
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.embed import file_html
import pandas as pd


def plotProfitFunction(df):
    pf = figure(plot_width=650, plot_height=400, title="PASS N Application")

    pf.xaxis.axis_label = "N application rate, kg/ha"
    pf.xaxis.axis_line_width = 3
    # pf.xaxis.axis_line_color = "red"
    pf.xaxis.axis_label_text_font_size = '16pt'
    pf.xaxis.axis_label_text_font = "times"
    pf.xaxis.axis_label_text_font_style = "bold"

    # change just some things about the y-axes
    pf.yaxis.axis_label = "Profit [$\ha]"
    # pf.yaxis.major_label_text_color = "orange"
    pf.yaxis.major_label_orientation = "vertical"
    pf.yaxis.axis_label_text_font = "times"
    pf.yaxis.axis_label_text_font_style = "bold"
    pf.yaxis.axis_label_text_font_size = '16pt'

    # pf.line([Npmax, Npmax, 0], [0, pmax, pmax], line_dash=[2, 2], line_color="orange")
    ExpProfSum = list(df.loc[:, 'EProf'])
    Napp = list(df['N Rate'])
    pf.line(Napp, ExpProfSum, line_color='green', line_width=2)
    #pf.line(Napp, df.loc[:, 'NRCNF'], line_dash=[2, 2], line_color="orange", line_width=2)
    EPYmax = max(ExpProfSum)  # maximum profit value
    EPNymax = Napp[ExpProfSum.index(EPYmax)]  # N rate at which maximum profit is achieved
    pf.line([EPNymax, EPNymax, 0], [0, EPYmax, EPYmax], line_dash=[2, 2], line_color="yellow", line_width=2)

    profitHtml = file_html(pf, CDN, "Profit Function")
    return profitHtml

def plotProfitClasses(df):
    # Plotting bokeh figure
    p = figure(plot_width=650, plot_height=400, title="PASS NApplication")

    # change just some things about the x-axes
    p.xaxis.axis_label = "N application rate, kg/ha"
    p.xaxis.axis_line_width = 3
    # p.xaxis.axis_line_color = "red"

    # change just some things about the y-axes
    p.yaxis.axis_label = "Yield, t\ha"
    # p.yaxis.major_label_text_color = "orange"
    p.yaxis.major_label_orientation = "vertical"
    # p.line(xparr, yparr, line_color='green', line_width=2)

    # html = file_html(p, CDN, "my plot")


    # NreturnYmax = max(Nreturns) # maximum profit at purticular N rate
    # NreturnNmax= Napp[Nreturns.index(NreturnYmax)] # N rate where maximum profit is  achieved
    # pf.line([NreturnNmax, NreturnNmax, 0], [0, NreturnYmax, NreturnYmax], line_dash=[2, 2], line_color="green",line_width=2)
    # Probability Model
    pr = figure(plot_width=650, plot_height=400, title="PASS N Application")
    pr.xaxis.axis_label = "N application rate, kg/ha"
    pr.xaxis.axis_line_width = 3
    # pf.xaxis.axis_line_color = "red"
    pr.xaxis.axis_label_text_font_size = '16pt'
    pr.xaxis.axis_label_text_font = "times"
    pr.xaxis.axis_label_text_font_style = "bold"

    # change just some things about the y-axes
    pr.yaxis.axis_label = "Probability percentage"
    # pf.yaxis.major_label_text_color = "orange"
    pr.yaxis.major_label_orientation = "vertical"
    pr.yaxis.axis_label_text_font = "times"
    pr.yaxis.axis_label_text_font_style = "bold"
    pr.yaxis.axis_label_text_font_size = '16pt'


    pr.line(df['N Rate'], df.loc[:, 'prob_1500'], line_color='black', line_width=2, line_dash=[2, 2])
    pr.line(df['N Rate'], df.loc[:, 'prob_1000'], line_color='black', line_width=2, line_dash=[3, 3])
    pr.line(df['N Rate'], df.loc[:, 'prob_2000'], line_color='black', line_width=2, line_dash=[4, 4])
    pr.line(df['N Rate'], df.loc[:, 'prob_2500'], line_color='black', line_width=3, line_dash=[5, 5])
    #show(pr)
    profitClassHtml = file_html(pr, CDN, "Profit Classes")

    return profitClassHtml