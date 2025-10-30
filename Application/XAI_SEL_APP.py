import streamlit as st


pg = st.navigation([
    st.Page("Page_1.py", title="Introduction"),
    st.Page("Page_2.py", title="Filtering methods"),
    st.Page("Page_3.py", title="Method selection using MCDM")])
pg.run()


st.set_page_config(  # Alternate names: setup_page, page, layout
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit".
	page_icon=None,  # String, anything supported by st.image, or None.
)





#if st.button("Show graph for each method separately", icon=":material/graph_6:", key = "graphs"):

from scipy.linalg import eigvals


#st.title("Saaty's Pairwise Comparison Method")

# Get criteria/alternatives from user


#for i in range(0, weights22.shape[0]):
 #   st.write('w(g' + str(i + 1) + '): ', round(weights22[i], 3))

# Consistency Ratio
#st.write('RC: ' + str(round(rc, 2)))
#if (rc > 0.10):
#    st.write('The solution is inconsistent, the pairwise comparisons must be reviewed')
#else:
#    st.write('The solution is consistent')


