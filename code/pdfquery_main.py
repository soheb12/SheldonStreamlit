import os
import tempfile
import configparser
import multiprocessing
import concurrent

import streamlit as st
from streamlit_chat import message

from pdfquery import PDFQuery

st.set_page_config(page_title="Sheldon")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            query_text = st.session_state["pdfquery"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((query_text, False))


def read_and_save_file():
    st.session_state["pdfquery"].forget()  # to reset the knowledge base
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # Create a directory to store uploaded files
    upload_dir = './uploaded_files'
    os.makedirs(upload_dir, exist_ok=True)

    for file in st.session_state["file_uploader"]:
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["pdfquery"].ingest(file_path)

        os.remove(file_path)


def main():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["file_summaries"] = {}  # add this line to initialize the summaries dictionary
        st.session_state["pdfquery"] = PDFQuery()

    st.header("Sheldon")

    st.subheader("Upload document(s)")
    st.file_uploader(
        "Upload document",
        type=["pdf", "docx"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    st.divider()


if __name__ == "__main__":
    main()