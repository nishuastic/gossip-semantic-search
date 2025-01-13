import requests
import streamlit as st

# Streamlit app
st.title("Semantic Search for VSD & Public")

# Input for search query
query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        # Send query to FastAPI backend
        response = requests.post("http://localhost:8000/search", json={"query": query})
        if response.status_code == 200:
            results = response.json()

            # Display results
            for result in results:
                st.write(f"### [{result['title']}]({result['url']})")
                st.write(result["summary"])
                st.write(f"**Category:** {result['category']}")
                st.write(f"**Published:** {result['published']}")
                st.write("---")
        else:
            st.error("Error fetching results from backend.")
    else:
        st.warning("Please enter a query.")
