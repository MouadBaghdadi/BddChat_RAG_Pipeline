import tkinter as tk
from dahakchat import DahakChatPipeline

def on_submit():
    user_query = query_entry.get()
    response = dahak_chat.ask_dahak(user_query)
    response_text.config(state=tk.NORMAL)
    response_text.delete("1.0", tk.END)
    response_text.insert(tk.END, response)
    response_text.config(state=tk.DISABLED)

def main():
    global dahak_chat

    # Initialize DahakChat pipeline
    dahak_chat = DahakChatPipeline()

    # Create main application window
    root = tk.Tk()
    root.title("BddChat")

    # Create GUI elements
    query_label = tk.Label(root, text="Bonjour, c'est BddChat comment je peut vous aidez?")
    query_label.pack()

    global query_entry
    query_entry = tk.Entry(root, width=50)
    query_entry.pack()

    submit_button = tk.Button(root, text="questionne", command=on_submit)
    submit_button.pack()

    global response_text
    response_text = tk.Text(root, width=60, height=10, state=tk.DISABLED)
    response_text.pack()

    # Run the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()