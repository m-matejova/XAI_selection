import streamlit as st


pg = st.navigation([
    st.Page("Page_1.py", title="Introduction"),
    st.Page("Page_2.py", title="Filtering methods"),
    st.Page("Page_3.py", title="Method selection using MCDM")])
pg.run()


st.set_page_config( 
	layout="wide",  
	initial_sidebar_state="expanded",  
	page_title=None,  
	page_icon=None,  
)
