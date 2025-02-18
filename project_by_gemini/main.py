import streamlit as st

if "tasks" not in st.session_state:
    st.session_state.tasks = []

st.title("Todo List")

task = st.text_input("Enter Task")

if st.button("ADD TASK") and task:
    st.session_state.tasks.append(task)

for index, task in enumerate(st.session_state.tasks):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f"{index+1}). {task}")
    with col2:
        if st.button(f"❌", key=f"delete_{index}"):
            st.session_state.tasks.pop(index)
            st.experimental_rerun()  