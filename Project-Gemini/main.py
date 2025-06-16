import streamlit as st

if "todos" not in st.session_state:
    st.session_state.todos = []

st.title("Todo List")

task = st.text_input("Enter Task")

if st.button("Add task") and task:
    st.session_state.todos.append(task)

for index, todo in enumerate(st.session_state.todos):
    st.write(f"{index+1}. {todo}")

    if st.button("delete", key=index):
        st.session_state.todos.pop(index)
        st.experimental_rerun()