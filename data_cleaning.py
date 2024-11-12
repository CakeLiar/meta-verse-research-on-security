import streamlit as st
import plotly.express as px
import os
import pandas as pd
import plotly.offline as pyo

st.selectbox("Movement", options=('Archery', 'Bowling'))

df = pd.read_csv('dataset_without_normalization_cleaned/Archery/p1/session1/1.csv')

df.rename(columns={ df.columns[0]: "id" }, inplace=True)

st.write(df)

mss = 1

# Create a figure for RightControllerAnchor
fig_right = px.scatter_3d(
    df,
    x="RightControllerAnchor_pos_X",
    y="RightControllerAnchor_pos_Y",
    z="RightControllerAnchor_pos_Z",
    color="timestamp_ms",
    template='plotly_dark',
    animation_frame='id',
    color_continuous_scale="reds",
)

# Update the layout for the RightControllerAnchor figure
fig_right.update_layout(
    scene=dict(
        xaxis=dict(range=[20.2, 20.6]),
        yaxis=dict(range=[1.15, 1.66]),
        zaxis=dict(range=[28.35, 28.63]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
    ),
    updatemenus=[dict(type='buttons',
                      showactive=False,
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=mss, redraw=True), fromcurrent=True)]),
                               dict(label='Pause',
                                    method='animate',
                                    args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                                    )
                               ]
                      )]
)

st.plotly_chart(fig_right)

# Create a figure for LeftControllerAnchor
fig_left = px.scatter_3d(
    df,
    x="LeftControllerAnchor_pos_X",
    y="LeftControllerAnchor_pos_Y",
    z="LeftControllerAnchor_pos_Z",
    color="timestamp_ms",
    template='plotly_dark',
    animation_frame='id',
    color_continuous_scale="reds",
)

# Update the layout for the LeftControllerAnchor figure
fig_left.update_layout(
    scene=dict(
        xaxis=dict(range=[19.4, 20.5]),
        yaxis=dict(range=[1.18, 1.64]),
        zaxis=dict(range=[28.3, 29.1]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
    ),
    updatemenus=[dict(type='buttons',
                      showactive=False,
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=mss, redraw=True), fromcurrent=True)]),
                               dict(label='Pause',
                                    method='animate',
                                    args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                                    )
                               ]
                      )]
)

st.plotly_chart(fig_left)

# Create a figure for CenterEyeAnchor
fig_center = px.scatter_3d(
    df,
    x="CenterEyeAnchor_pos_X",
    y="CenterEyeAnchor_pos_Y",
    z="CenterEyeAnchor_pos_Z",
    animation_frame='id',
    color="timestamp_ms",
    template='plotly_dark',
    color_continuous_scale="reds",
)

# Update the layout for the CenterEyeAnchor figure
fig_center.update_layout(
    scene=dict(
        xaxis=dict(range=[20, 20.5]),
        yaxis=dict(range=[1.73, 1.79]),
        zaxis=dict(range=[28.2, 28.5]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
    ),
    updatemenus=[dict(type='buttons',
                      showactive=False,
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=mss, redraw=True), fromcurrent=True)]),
                               dict(label='Pause',
                                    method='animate',
                                    args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                                    )
                               ]
                      )]
)

st.plotly_chart(fig_center)
