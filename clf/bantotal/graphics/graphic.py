#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing library for manipulation and exploration of datasets.
import numpy as np

# Importing the functions auxiliary
from functions import functions_aux
# Importing the transformed functions
from custom_transformers import transformer

# Importing necessary libraries for plotting interactive graphs with plotly.
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Importing libraries needed for matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
#import joypy
#-------------------------------------------------------

""""A named constant is a name that represents a value that cannot be 
changed during the program's execution."""

# Using Global Constants Defining Named Constants
HEIGHT = 720
WIDTH = 1200
TITLE_FONT={'size':20, 'family': 'Times New Roman',}
TITLE_X=0.5
FONT_COLOR = "#000000"
#-------------------------------------------------------

# Creating a function to view missing data
def plot_missing_values(df, template):

    # Function composition
    missing_df = functions_aux.missing_values(df)

    # Creates the figure with horizontal orientation
    fig = go.Figure(go.Bar(x=missing_df['percent_missing'], y=missing_df['column'], orientation="h"))
    # Format Layout
    fig.update_layout(title_text='<b>Percentage of missing values in Features</b>',
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height= HEIGHT,
                      width= WIDTH,
                      xaxis=dict(ticksuffix="%"),
                      yaxis_title="Variável(is) ",
                      template= template
                      )
    # white space adjustment
    fig.update_yaxes(ticksuffix="   ")

    # Return the graph
    return fig.show()

# Creating an function for visualize Principal Component Analysis (PCA) of your high-dimensional data
"""
Credit:
https://plotly.com/python/pca-visualization/
https://community.plotly.com/t/set-pca-loadings-aka-arrows-in-a-3d-scatter-plot/72905
#https://community.plotly.com/t/set-pca-loadings-aka-arrows-in-a-3d-scatter-plot/72905
"""
def plot_pca(
    df,features,
    template,
    arrowsize = 1,
    arrowhead = 1,
    arrowscale = 6,
    ):
    """
    Compute PCA function composition
    Return the PCA calculation, components and load variable
    """
    pca, components,loadings = transformer.computePCA_v2(df,features)

    # Create the figure
    fig = px.scatter(components, x=0, y=1)
    for i, feature in enumerate(features):
        fig.add_annotation(
            ax=0, ay=0,
            axref="x", ayref="y",
            x=loadings[i, 0]*arrowscale,
            y=loadings[i, 1]*arrowscale,
            showarrow=True,
            arrowsize=arrowsize,
            arrowhead=arrowhead,
            xanchor="right",
            yanchor="top"
        )
        fig.add_annotation(
            x=loadings[i, 0]*arrowscale,
            y=loadings[i, 1]*arrowscale,
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            yshift=5,
        )
    fig.update_layout(title='<b>Total explained variance PC1 + PC2: {}%</b>'.format(round(pca.explained_variance_ratio_[0:2].cumsum()[-1]*100,2)),
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      template= template,
    )
    # Return fig
    fig.show()

# Create function for 3D scatter plot
def plot_3D_pca(

    df,features,
    template,
    arrowsize = 1,
    arrowhead = 1,
    arrowscale = 6,
    ):
    """
    Compute PCA function composition
    Return the PCA calculation, components and load variable
    """
    pca, components,loadings = transformer.computePCA_v2(df,features)

    # Create the figure
    fig = px.scatter_3d(components, x=0, y=1, z=2)
    for i, feature in enumerate(features):
        fig.update_layout(
            scene=dict(
                annotations=[
                    dict(
                        # ax=0, ay=0,
                        showarrow=True,
                        arrowsize=arrowsize,
                        arrowhead=arrowhead,
                        x=loadings[i, 0] * arrowscale,
                        y=loadings[i, 1] * arrowscale,
                        z=loadings[i, 2] * arrowscale,
                        xanchor="center",
                        yanchor="bottom",
                        text=feature,
                        yshift=5,
                    )
                    for i, feature in enumerate(features)]
            ),
        title = '<b>Total explained variance PC1 + PC2 + PC3: {}%</b>'.format(round(pca.explained_variance_ratio_[0:3].cumsum()[-1]*100, 2)),
        title_x=TITLE_X,
        title_font=TITLE_FONT,
        font_color=FONT_COLOR,
        height = HEIGHT,
        width = WIDTH,
        template = template,
        )
    # Return fig
    fig.show()

# Create function for correlation matrix plot
def plot_matrix_corr(df, template):

    # Calculate the correlation matrix an eliminates negative values
    corr = df.corr().abs()
    # Select upper triangle of correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Mask matrix correlation
    corr = corr.mask(mask)

    # Create a figure
    fig = ff.create_annotated_heatmap(
        z=corr.to_numpy().round(2),
        x=list(corr.index.values),
        y=list(corr.columns.values),
        xgap=3, ygap=3,
        zmin=-1, zmax=1,
        colorscale='rdbu',
        colorbar_thickness=30,
        colorbar_ticklen=3,
        showscale= True,
    )
    # Format Layout
    fig.update_layout(title_text='<b>Correlation Matrix (cont. features)<b>',
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color =FONT_COLOR,
                      margin=dict(l=200,b=200),
                      height = HEIGHT,
                      width = WIDTH,
                      xaxis_showgrid=False,
                      xaxis={'side': 'bottom'},
                      yaxis_showgrid=False,
                      xaxis_zeroline=False,
                      yaxis_zeroline=False,
                      yaxis_autorange='reversed',
                      paper_bgcolor=None,
                      template=template,
                      )
    #fig.update_xaxes(tickangle=45,matches=None, showticklabels=True, visible=True,)
    # Return fig
    return fig

# Create an function for histograma using matplotlib
def plot_hist(df,title):
    # Creating histogram for each variable in the dataset
    df.hist(bins=20, rwidth=0.9, figsize=(14, 16))

    # Added title to chart
    plt.suptitle(title, size=18)
    plt.show()

def plot_histograma(df, x, y, color, title, x_title, y_title, legend, template):
    # Create plot are
    fig = px.histogram(df, x=x, y=y, color=color)

    # Format Layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title=x_title,
                      yaxis_title=y_title,
                      yaxis=dict(tickformat=".0f"),
                      legend_title_text=legend,
                      template=template
                      )
    # Histogram overlay
    # fig.update_layout(barmode = 'overlay')
    fig.update_traces(opacity=0.74)

    # Return the graph
    return fig.show()

# Creating a scatter function for regression analysis
def plot_scatter(df, x, y, color, title, x_title, y_title, template):

    # Create a figure
    fig = px.scatter(data_frame=df,
                     x=x,
                     y=y,
                     color=color,
                     symbol="genre",
                     size='global_sales',
                     marginal_y="violin",
                     marginal_x="box",
                     trendline="ols",
    )
    # Format Layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      height=HEIGHT,
                      width=WIDTH,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      xaxis_title=x_title,
                      yaxis_title=y_title,
                      xaxis=dict(ticksuffix="M"),
                      yaxis=dict(ticksuffix="M"),
                      template=template,
    )
    fig.show()

# Z Score calculation plot function
def plot_calc_zscore(
    df,categorical,
    title_bar,title_hist,title_dist,y_title_bar,
    template):

    # Creates the figure with horizontal orientation
    fig = go.Figure(go.Bar(y = df[categorical], x = df["z_score"], orientation = "h",marker_color=df["colors"]))

    # Format Layout
    fig.update_layout(title_text=title_bar,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title = "Z Score",
                      yaxis_title = y_title_bar,
                     template= template)
    fig.show()

    # Create the second graphic of the figure
    fig = px.histogram(df, x=df["z_score"],
                          marginal = "box", # or violin, rug
                          color=df["colors"],
                          color_discrete_sequence = ["red","green"],nbins=30)
    # Format Layout
    fig.update_layout(title_text=title_hist,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title = "Z Score",
                      yaxis_title = "Frenquecy",
                      template=template)
    fig.show()

    # Data for the histogram
    hist_data = [df["z_score"]]
    # Labels
    group_labels = ["distribution"]
    # Colors
    colors = [df['colors']]
    # Graph with flexibility of a univariate distribution of observations.
    fig = ff.create_distplot(hist_data, group_labels, colors = colors)
    # Format Layout
    fig.update_layout(title_text=title_dist,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      yaxis=dict(tickformat=".0%"),
                      template=template)
    fig.show()

# Create pareto chart using the 80/20 rule
def plot_pareto(df, categorical, numeric, title, template):

    trace1 = go.Bar(
        x=df[categorical],
        y=df[numeric],
        name="Platform",
        marker=dict(color='LightSeaGreen'),
        text=df[numeric], textposition='outside', textfont_size=20, textangle=-45, cliponaxis=False,
    )
    trace2 = go.Scatter(
        x=df[categorical],
        y=df['Fr'],
        mode='lines+markers',
        name='Cumulative frequency',
        marker=dict(color='orange'),
        yaxis='y2'
    )
    # Create  a figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add graphics to subplots
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, secondary_y=True)
    # Format Layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      yaxis_title="Total sales (in millions)",
                      yaxis=dict(ticksuffix="M"),
                      yaxis2=dict(ticksuffix="%"),
                      template=template)
    fig.show()

# Creating a function for frequency distribution
# https://plotly.com/python/horizontal-bar-charts/
def plot_bar_with_line(df, categorical, percent, sales_amount, title, template):
    # Ordering
    df = df.sort_values([percent])

    # Capturing the names convert the unique list
    names = df[categorical].unique().tolist()
    # names = [cols for cols in df.columns]

    # Creating two subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                        shared_yaxes=False, vertical_spacing=0.001)

    fig.append_trace(go.Bar(
        x=df[percent],
        y=names,
        marker=dict(
            color='rgba(50, 171, 96, 0.6)',
            line=dict(
                color='LightSeaGreen',
                width=2),
        ),
        name='Total Sales by gender in percentage terms',
        orientation='h',
    ), 1, 1)
    fig.append_trace(go.Scatter(
        x=df[sales_amount], y=names,
        mode='lines+markers',
        line_color='DarkBlue',
        name='Total Sales by gender',
    ), 1, 2)
    fig.update_layout(
        title_text=title,
        title_x=TITLE_X,
        title_font=TITLE_FONT,
        font_color=FONT_COLOR,
        height=HEIGHT,
        width=WIDTH,
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            domain=[0, 0.85],
            ticksuffix=" ",
        ),
        yaxis2=dict(
            showgrid=False,
            showline=True,
            showticklabels=False,
            linecolor='MidnightBlue',
            linewidth=2,
            domain=[0, 0.85],
        ),
        xaxis=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
            ticksuffix="%",
            domain=[0, 0.42],
            side='top',
        ),
        xaxis2=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
            ticksuffix="M",
            domain=[0.47, 1],
            side='top',
        ),
        legend=dict(x=0.029, y=1.038, font_size=10),
        margin=dict(l=100, r=20, t=70, b=70),
        template=template,
    )
    annotations = []
    freq_relative = np.round(df[percent], 2)
    sa = np.rint(df[sales_amount])

    # Adding labels
    for ydn, yd, xd in zip(sa, freq_relative, names):
        # labeling the scatter savings
        annotations.append(dict(xref='x2', yref='y2',
                                y=xd, x=ydn + 50,
                                text='{:,}'.format(ydn) + 'M',
                                font=dict(family='Arial', size=12,
                                          color='rgb(12, 0, 128)'),
                                showarrow=False))
        # labeling the bar net worth
        annotations.append(dict(xref='x1', yref='y1',
                                y=xd, x=yd + 3,
                                text=str(yd) + '%',
                                font=dict(family='Arial', size=12,
                                          color='rgb(50, 171, 96)'),
                                showarrow=False))
    fig.update_layout(annotations=annotations, height=HEIGHT, width=WIDTH, template=template)
    fig.show()

#  Function to graph the predictions of the gamma distribution
def plot_dist_gamma(df, title, template):
    # Creating labels
    group_labels = ['Predictions', 'Actual']
    # Create the figure
    fig = ff.create_distplot([df.Predictions, df.Actual],
                             group_labels,
                             show_rug=True,  # rug
                             show_hist=False,  # hist
                             )
    # Format layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title='Probability Distribution',
                      yaxis_title='Density',
                      showlegend=True,
                      template="ygridoff",
                      )
    fig.show()

# Creating graphs for time series analysis
def ts_series(df, *args, title, template, ):
    # Create the figure
    fig = px.line(df, y=[arg for arg in args])

    # Changing the names in the legend without changing the font using a dict
    newnames = {'na_sales': 'América do Norte', 'eu_sales': 'Europa', 'jp_sales': 'Japão',
                'other_sales': 'Resto do Mundo'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                          )
                       )
    # Format eixos x
    fig.update_xaxes(tickangle=-45, nticks=35)
    fig.update_traces(hoverinfo='text+name', mode='lines+markers')
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      legend_title="Continentes",
                      xaxis_title="years",
                      yaxis_title="Total sales (in millions)",
                      template=template)

    # Show plot
    fig.show()

    # Create Density Chart
    # Creating labels
    group_labels = ["América do Norte", "Europa", "Japão", "Resto do Mundo"]

    # Create the figure
    fig = ff.create_distplot([df["na_sales"], df["eu_sales"], df["jp_sales"], df["other_sales"]],
                             group_labels,
                             show_hist=False,
                             show_rug=False,
                             )
    # Format layout
    fig.update_layout(title_text="<b>Estimativa de Densidade das vendas de video games</b>",
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title="Total sales (in millions)",
                      yaxis_title='Density',
                      showlegend=True,
                      legend_title="Continentes",
                      xaxis=dict(ticksuffix="M"),
                      template=template
                      )
    fig.show()

# Create the graph of density
def density(df, title, template):
    # Creating labels
    group_labels = ["América do Norte", "Europa", "Japão", "Resto do Mundo"]

    # Create the figure
    fig = ff.create_distplot([df["na_sales"], df["eu_sales"], df["jp_sales"], df["other_sales"]],
                             group_labels,
                             show_hist=False,show_rug=False,
                             )
    # Format layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title="Total sales (in millions)",
                      yaxis_title='Density',
                      showlegend=True,
                      legend_title="Continentes",
                      xaxis=dict(ticksuffix="M"),
                      template=template
                      )
    fig.show()

def plot_joy(df, categorical,title, *args):
    # Setting the font scale
    sns.set(font_scale=1.0)
    # Alter theme
    sns.set(style="white")
    # Creating the representation in the plot area
    #plt.figure(figsize=(16, 10), dpi=80)
    fig, axes = joypy.joyplot(df, column=[arg for arg in args],
                               by=[categorical], ylim='own',
                               legend = True,
                               xlabels=True,
                               figsize=(16,16))
    # Decoration
    plt.title(title, fontsize=20)
    plt.xlabel('Total sales by platform (in millions)',fontsize =16)
    plt.xticks(rotation=45)
    return plt.show()

# Function for graphing confidence intervals
def plot_ci(df, numeric, title, template):
    # Create the figure
    fig = go.Figure([
        go.Scatter(
            name='Avg Sales',
            x=df[numeric],
            y=round(df['mean'], 2),
            mode='lines+markers',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='95% CI Upper',
            x=df[numeric],
            y=round(df['ci_upper'], 2),
            mode='lines+markers',
            marker=dict(color='LightSeaGreen'),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Lower',
            x=df[numeric],
            y=round(df['ci_lower'], 2),
            marker=dict(color='#d62728'),
            line=dict(width=0),
            mode='lines+markers',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title='Year',
                      yaxis_title='Avg Sales',
                      hovermode='x',
                      template=template
                      )
    fig.update_xaxes(tickangle=-45, nticks=40)
    fig.update_yaxes(rangemode='tozero')
    fig.show()

# Create function for graph de pie
def plot_pie(df, categorical1, categorica2, numeric, title,title_pie1, title_pie2):

    # Create the subplots
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'domain'}, {'type':'domain'}],])
    fig.add_trace(
        go.Pie(
            labels=df[categorical1],
            title=title_pie1,
            title_font=TITLE_FONT,
            values=df[numeric],
            hole=0.7,
            ),  col=1, row=1,
        )
    fig.update_traces(
        hoverinfo='label+value',
        textinfo='label+percent',
        textfont_size=12,
        )

    fig.add_trace(
        go.Pie(
            labels=df[categorica2],
            title=title_pie2,
            title_font=TITLE_FONT,
            values=df[numeric],
            hole=0.5,
            ),  col=2, row=1,
        )
    fig.update_traces(
        hoverinfo='label+value',
        textinfo='label+percent',
        textfont_size=12,
        )
    fig.layout.update(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      showlegend=False,
                      template=None,
                     )
    fig.show()

def plot_preditction(y_test,y_pred,title,template):

    # Create the figures
    fig = go.Figure([
        #go.Scatter(y=y_train, name='train', mode='markers'),
        go.Scatter(y=y_test, name='test', mode='markers'),
        go.Scatter(y=y_pred, name='prediction')
    ])

    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height= HEIGHT,
                      width= WIDTH,
                      xaxis_title='Total sales per Week',
                      yaxis_title='Forecast',
                      template= template,
                     )
    fig.show()

def plot_classes(df, categorical, title, legend, template):
    # creating the figure
    fig = go.Figure([go.Pie(labels=df[categorical])])

    # Interatividade
    fig.update_traces(hoverinfo='label+percent',
                      textinfo='value+percent',
                      textfont_size=15,
                      insidetextorientation='radial'
                      )
    # Format Layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      legend_title_text=legend,
                      template=template
                      )
    # Return the graph
    return fig.show()

def plot_crosstab(df, title, x_title, y_title, legend, template):
    # Create a one list empty
    data = []
    for x in df.columns:
        data.append(go.Bar(name=str(x), x=df.index, y=df[x]))
    # Create plot area
    fig = go.Figure(data)

    # Format Layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title=x_title,
                      yaxis_title=y_title,
                      legend_title_text=legend,
                      barmode='group',
                      template=template
                      )
    # Return the graph
    return fig.show()

def plot_facet(df, x, y, color, col, title,y_title, legend, template):
    # Create the facet
    fig = px.bar(df, x=x, y=y,color=color,facet_col=col,barmode="group")
    # Format Layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=HEIGHT,
                      width=WIDTH,
                      xaxis_title="",xaxis2_title="",
                      yaxis_title=y_title,
                      yaxis=dict(tickformat=".0f"),
                      yaxis2=dict(tickformat=".0f"),
                      legend_title_text=legend,
                      showlegend=True,
                      template=template
                      )
    fig.update_xaxes(tickangle=45, matches=None, showticklabels=True, visible=True,)
    fig.update_yaxes(matches=None,showticklabels=True,)
    fig.for_each_yaxis(lambda y: y.update(title=''))
    for i, label in enumerate(['<b>Negado</b>', '<b>Autorizado</b>']):
        fig.layout.annotations[i]['text'] = label

    # Return the graph
    return fig.show()


def plot_facet2(df, x, y, color, col, row, title, y_title, legend, template):
    # Create the facet
    fig = px.bar(df, x=x, y=y, color=color, facet_row=row, facet_col=col, barmode="stack")
    # Format Layout
    fig.update_layout(title_text=title,
                      title_x=TITLE_X,
                      title_font=TITLE_FONT,
                      font_color=FONT_COLOR,
                      height=1000,
                      width=WIDTH,
                      xaxis_title="", xaxis6_title="",
                      yaxis=dict(tickformat=".0f"),
                      legend_title_text=legend,
                      showlegend=True,
                      template=template
                      )
    # Loop for rename columns
    for i, label in enumerate(['1x parc', '2x parc', '3x parc', '4x parc', '5x parc', '6x par']):
        fig.layout.annotations[i]['text'] = label

    fig.update_xaxes(tickangle=45, matches=None, showticklabels=True, visible=True, )
    fig.for_each_yaxis(lambda y: y.update(title=''))
    fig.add_annotation(x=-0.05, y=0.8,
                       text=y_title, textangle=-90,
                       xref="paper", yref="paper",
                       font=dict(size=18), )
    # Return the graph
    return fig.show()
"""
References
https://plotly.com/python/
https://community.plotly.com/t/set-pca-loadings-aka-arrows-in-a-3d-scatter-plot/72905
https://www.kaggle.com/code/desalegngeb/plotly-guide-customize-for-better-visualizations
"""
