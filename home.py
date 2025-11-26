import streamlit as st
import csv
import os
import subprocess

# Function to check if user already exists
def user_exists(username):
    with open('users.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['username'] == username:
                return True
    return False

# Function to create an account
def signup(username, password):
    with open('users.csv', 'a', newline='') as csvfile:
        fieldnames = ['username', 'password']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'username': username, 'password': password})

# Function to authenticate user login
def login(username, password):
    with open('users.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['username'] == username and row['password'] == password:
                return True
    return False

# Main Streamlit app
def main():
    st.title("Simple Sign Up and Login")

    page = st.sidebar.selectbox("Choose a page", ["Home", "Sign Up", "Login"])

    if page == "Home":
        st.header("Welcome to the Simple Sign Up and Login App")
        st.write("Select a page from the sidebar to get started.")

    elif page == "Sign Up":
        st.header("Sign Up")
        username = st.text_input("Enter your username:")
        password = st.text_input("Enter your password:", type="password")

        if st.button("Sign Up"):
            if user_exists(username):
                st.error("Username already exists. Please choose a different one.")
            else:
                signup(username, password)
                st.success("Sign up successful! You can now login.")

    elif page == "Login":
        st.header("Login")
        username = st.text_input("Enter your username:")
        password = st.text_input("Enter your password:", type="password")

        if st.button("Login"):
            if login(username, password):
                st.success(f"Welcome, {username}!")
                # Open a new Streamlit app or web page using subprocess
                subprocess.Popen(["streamlit", "run", "app.py"])
            else:
                st.error("Login failed. Please check your username and password.")

if __name__ == '__main__':
    if not os.path.exists('users.csv'):
        with open('users.csv', 'w', newline='') as csvfile:
            fieldnames = ['username', 'password']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    main()
