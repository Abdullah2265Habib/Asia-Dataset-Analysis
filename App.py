#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#######################
# Page configuration
st.set_page_config(
    page_title="Asia Population Dashboard",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################
# Load data
df = pd.read_csv('AsiaPopulation2020.csv')
df['Population'] = df['Population'].astype('int64')
df['NetChange'] = df['NetChange'].astype('int64')
df['Density'] = df['Density'].astype('int64')
df['LandArea'] = df['LandArea'].astype('int64')
df['Migrants'] = df['Migrants'].fillna(0).astype('int64')

#######################
# Sidebar
with st.sidebar:
    st.title('🌏 Asia Population Dashboard')

    metric_options = {
        'Population': 'Population',
        'Density': 'Density (per km²)',
        'YearlyChange': 'Yearly Change (%)',
        'NetChange': 'Net Change',
        'FertRate': 'Fertility Rate',
        'MedAge': 'Median Age',
        'UrbanPop': 'Urban Population (%)'
    }

    selected_metric = st.selectbox('Select a metric', list(metric_options.keys()), format_func=lambda x: metric_options[x])
    selected_color_theme = st.selectbox('Select a color theme', ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis'])
    selected_sort = st.selectbox('Sort countries', ['Highest to Lowest', 'Lowest to Highest'])

    st.markdown("### Scatter Plot Inputs")
    scatter_dep = st.selectbox('Dependent (Y)', list(metric_options.keys()), index=0, key='scatter_dep', format_func=lambda x: metric_options[x])
    scatter_indep = st.selectbox('Independent (X)', list(metric_options.keys()), index=1, key='scatter_indep', format_func=lambda x: metric_options[x])

    st.markdown("### Regression Inputs")
    reg_dep = st.selectbox('Dependent (Y)', list(metric_options.keys()), index=0, key='reg_dep', format_func=lambda x: metric_options[x])
    reg_indep = st.selectbox('Independent (X)', list(metric_options.keys()), index=1, key='reg_indep', format_func=lambda x: metric_options[x])
    linear_prediction = st.number_input("Enter Independent Value for Prediction:", key='reg_input')


#######################
# Helper functions
def format_number(num):
    if abs(num) > 1000000:
        return f'{num/1000000:.1f}M'
    elif abs(num) > 1000:
        return f'{num/1000:.0f}K'
    return str(num)

def make_choropleth(input_df, metric, color_theme):
    fig = px.choropleth(
        input_df,
        locations="Country",
        locationmode="country names",
        color=metric,
        hover_name="Country",
        hover_data={metric: True, "Population": True},
        color_continuous_scale=color_theme,
        scope="asia",
        labels={metric: metric_options[metric]}
    )
    fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0), height=500)
    return fig

def make_barchart(input_df, metric, color_theme):
    sort_order = False if selected_sort == 'Highest to Lowest' else True
    sorted_df = input_df.sort_values(by=metric, ascending=sort_order).head(10)
    chart = alt.Chart(sorted_df).mark_bar().encode(
        x=alt.X(metric, title=metric_options[metric]),
        y=alt.Y('Country', sort=alt.EncodingSortField(field=metric, order='descending' if sort_order else 'ascending')),
        color=alt.Color(metric, scale=alt.Scale(scheme=color_theme), legend=None),
        tooltip=['Country', metric]
    ).properties(height=400)
    return chart

def make_piechart(input_df, metric, color_theme_name):
    top_countries = input_df.sort_values(by=metric, ascending=False).head(5)
    color_seq = getattr(px.colors.sequential, color_theme_name.capitalize(), px.colors.sequential.Blues)
    fig = px.pie(
        top_countries,
        names='Country',
        values=metric,
        title=f"Top 5 Countries by {metric_options[metric]}",
        color_discrete_sequence=color_seq
    )
    fig.update_layout(template='plotly_dark', height=300, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    fig.update_traces(textposition='inside', textinfo='label+percent')
    return fig

def make_histogram(DF, column, color_theme):
    return alt.Chart(DF).mark_bar().encode(
        x=alt.X(column, bin=True, title=metric_options[column]),
        y='count()',
        color=alt.Color(column, bin=True, scale=alt.Scale(scheme=color_theme), legend=None),
        tooltip=['Country', column]
    ).interactive()

def make_scatter(DF, dep, indep):
    scatter = alt.Chart(DF).mark_circle(size=100).encode(
        x=alt.X(indep, title=metric_options[indep]),
        y=alt.Y(dep, title=metric_options[dep]),
        color=alt.value('white'),
        tooltip=['Country', indep, dep]
    )
    line = alt.Chart(DF).mark_line().encode(
        x=indep,
        y=dep,
        color=alt.value('red')
    )
    return (scatter + line).interactive()

def make_linear_regression(DF, dep, indep, input_val=None):
    scatter = alt.Chart(DF).mark_circle(size=100).encode(
        x=alt.X(indep, title=metric_options[indep]),
        y=alt.Y(dep, title=metric_options[dep]),
        color=alt.value('white'),
        tooltip=['Country', indep, dep]
    )
    reg_line = alt.Chart(DF).transform_regression(indep, dep).mark_line(color='red').encode(
        x=indep,
        y=dep
    )
    return (scatter + reg_line).interactive()

#######################
# Tabs
tab1, tab2 = st.tabs(["Dashboard", "Advanced Analysis"])

#######################
# TAB 1: Dashboard
with tab1:
    col = st.columns((1.5, 4.5, 2), gap='medium')

    with col[0]:
        st.markdown('#### Extremes')
        max_country = df.loc[df[selected_metric].idxmax()]
        min_country = df.loc[df[selected_metric].idxmin()]
        st.metric(label=f"Highest: {max_country['Country']}", value=format_number(max_country[selected_metric]))
        st.metric(label=f"Lowest: {min_country['Country']}", value=format_number(min_country[selected_metric]))

    with col[1]:
        st.markdown(f'#### {metric_options[selected_metric]} in Asia')
        st.plotly_chart(make_choropleth(df, selected_metric, selected_color_theme), use_container_width=True)
        st.markdown(f'#### Top/Bottom Countries: {metric_options[selected_metric]}')
        st.altair_chart(make_barchart(df, selected_metric, selected_color_theme), use_container_width=True)

    with col[2]:
        st.markdown('#### Country Rankings')
        sort_order = False if selected_sort == 'Highest to Lowest' else True
        sorted_df = df.sort_values(by=selected_metric, ascending=sort_order)
        st.dataframe(sorted_df[['Country', selected_metric]].head(10), hide_index=True, height=400)

    st.markdown('#### Top 5 Countries')
    st.plotly_chart(make_piechart(df, selected_metric, selected_color_theme), use_container_width=True)

    st.markdown('#### Histogram')
    st.altair_chart(make_histogram(df, selected_metric, selected_color_theme), use_container_width=True)

    st.markdown('#### Scatter Plot')
    st.altair_chart(make_scatter(df, scatter_dep, scatter_indep), use_container_width=True)

#######################
# TAB 2: Advanced Analysis
with tab2:
    st.markdown("### Correlation Matrix")
    corr_df = df[['Population', 'Density', 'FertRate', 'MedAge', 'UrbanPop']].corr().round(2)
    st.plotly_chart(
        px.imshow(corr_df, text_auto=True, color_continuous_scale=selected_color_theme),
        use_container_width=True
    )

    st.markdown("#### Linear Regression Model")
    x = df[reg_indep]
    y = df[reg_dep]
    x_mean = x.mean()
    y_mean = y.mean()
    b = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean)**2).sum()
    a = y_mean - b * x_mean
    st.latex(f"Y = {a:.2f} + {b:.2f} \\cdot X")

    pred_val = b * linear_prediction + a
    st.success(f"Predicted {metric_options[reg_dep]} for {linear_prediction} {metric_options[reg_indep]} is **{pred_val:.2f}**")

    st.markdown("### Regression Plot")
    st.altair_chart(make_linear_regression(df, reg_dep, reg_indep), use_container_width=True)
