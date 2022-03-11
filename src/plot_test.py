
import numpy as np
import pandas as pd
import plotly.offline as py                    #保存图表，相当于plotly.plotly as py，同时增加了离线功能
# py.init_notebook_mode(connected=True)          #离线绘图时，需要额外进行初始化
import plotly.graph_objs as go                 #创建各类图表
import plotly.figure_factory as ff
import plotly.express as px
import time

def display(train_loss, evaluate_scores):
    # print(np.arange(len(train_loss)))
    # print(np.array(train_loss))
    # print(np.arange(len(evaluate_scores)))
    # print(np.array(evaluate_scores))
    train_loss_trace = go.Scatter(
        x = np.arange(len(train_loss)),
        y = np.array(train_loss),
        mode = 'lines + markers',
        name = 'line + marker'
    )
    evaluate_loss_trace = go.Scatter(
        x = np.arange(len(evaluate_scores)),
        y = np.array(evaluate_scores),
        mode = 'lines + markers',
        name = 'line + marker'
    )

    loss_data = [train_loss_trace, evaluate_loss_trace]

    loss_layout = dict(title='loss ', xaxis=dict(title='step'), yaxis=dict(title='value'))
    loss_fig = dict(data=loss_data, layout=loss_layout)
    py.iplot(loss_fig, filename='loss-line')

display_df = pd.DataFrame(columns=['step', 'train_loss', 'evaluate_scores'])

for i in range(10):

    display_df.loc[i] = [i, i+1, 3]
    # print(display_df)
    # display(train_loss, evaluate_scores)

    # print("sleep 1 s")
    # time.sleep(1)
print(display_df)
fig = px.scatter(display_df,
             y="train_loss",
             x="step",
             animation_frame="step"
        )
fig.update_layout(width=1000,
                  height=800,
                  xaxis_showgrid=False,
                  yaxis_showgrid=False,
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  title_text='Evolution of Natural Disasters',
                  showlegend=False)
fig.update_xaxes(title_text='Number of Deaths')
fig.update_yaxes(title_text='')
fig.show()