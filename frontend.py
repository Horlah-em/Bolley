import os
import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Import the async function
from newmain import async_run_llm_agent

# Streamlit UI components
selected = option_menu(
    menu_title="🏀 Bolley",
    options=["Home", "About", "Contacts"],
    icons=["house", "book", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

async def get_response(query):
    return await async_run_llm_agent(query)

def main():
    if selected == "Home":
        prompt = st.chat_input("Make your request")

        if prompt:
            with st.spinner("Generating response..."):
                try:
                    # Run the async function using asyncio.run
                    generated_response = asyncio.run(get_response(prompt))
                    formatted_response = generated_response.get("output", "An error occurred.")
                except Exception as e:
                    formatted_response = f"An error occurred: {e}"

                st.session_state["user_prompt_history"].append(prompt)
                st.session_state["chat_answers_history"].append(formatted_response)

            if st.session_state["chat_answers_history"]:
                for generated_answer, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
                    message(user_query, is_user=True)
                    message(generated_answer)

    if selected == "Contacts":
        st.title(f"{selected}")
        st.text("Contact page.")
        with st.chat_message("AI"):
            st.markdown("About: ")
            st.markdown("Name: Emmanuel Okunlola")
            st.markdown("Email: emmanuelokunlola225@gmail.com")

    if selected == "About":
        st.text("About page.")
        with st.chat_message("AI"):
            st.markdown("About: ")
            st.markdown("A conversational bot to help basketball fans to get the latest updates on basketball fixtures, players stats, club stats and latest information about the NBA")
            st.markdown("")

if __name__ == "__main__":
    main()
