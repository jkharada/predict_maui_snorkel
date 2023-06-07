import streamlit as st
from datetime import date, timedelta
from snorkel_background import update_results
from plotting import plot_error

st.set_page_config(layout="wide")

nw, df_results = update_results()
k = 3.5
s = 6.0


def assign_color(score):
    if score <= 3.5:
            return 'red'
    elif score > 6.5:
            return 'green'
    else:
            return 'orange'

nw_color = assign_color(nw)
#k_color = assign_color(k)
#s_color = assign_color(s)

st.title(f"Maui Snorkel Prediction for {date.today():%B %d, %Y }")
st.write('The Snorkel Store does not endorse the results of this project as a scientifically accurate predictor of Maui ocean conditions.')
col1, col2, col3 = st.columns(3)

st.markdown("""---""")

st.title(f"Northwest: :{nw_color}[{nw:.1f}]")
#st.header("Northwest")
st.write("(Napili, Kapalua, Honolua)")
#st.write(df_results)
#st.write(plot_error(df_results))   
st.altair_chart(plot_error(df_results), theme=None)
    

# with col2:
#     st.title(f":{k_color}[{k:.1f}]")
#     st.header("Ka'anapali")
#     st.write("Black Rock, Kahekili (Airport Beach)")

# with col3:
#     st.title(f":{s_color}[{s:.1f}]")
#     st.header("South Shore")
#     st.write("Olowalu, Kihei, Wailea, Makena")
