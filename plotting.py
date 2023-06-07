import altair as alt
import pandas as pd

def plot_error(df):
    df_error = df[['Dates','Actual','Prediction']].melt('Dates', var_name='Label', value_name='Scores')
    df_error['Scores'] = df_error['Scores'].apply(pd.to_numeric)
      
    error_plot = alt.Chart(df_error).mark_point().encode(
        x = alt.X('Dates', axis=alt.Axis(labelAngle=-45)),
        y = 'Scores',
        color = alt.Color('Label').scale(scheme='category10')
    ).configure_axis(
        labelFontSize = 18,
        titleFontSize = 18
    ).configure_legend(
        labelFontSize = 18,
        titleFontSize = 18
    ).configure_point(
        size = 200,
        filled=True
    ).properties(
        width = 650
    )

    return error_plot
