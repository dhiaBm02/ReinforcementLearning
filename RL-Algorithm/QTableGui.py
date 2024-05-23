import tkinter as tk
from tkinter import font
from reinforcementLearningWithGui import load_q_table
# Example Q-table
q_table = load_q_table('QTable.json')

# Initialize Tkinter window
root = tk.Tk()
root.title("Q-Table Display")

# Customize fonts and colors
header_font = font.Font(family="Helvetica", size=12, weight="bold")
q_value_font = font.Font(family="Helvetica", size=10)
bg_color = "#f0f0f0"
header_color = "#c0c0c0"

# Create a frame for the Q-table
frame = tk.Frame(root, bg=bg_color)
frame.pack(padx=10, pady=10, fill="both", expand=True)


# Function to create Q-value labels for each state
def create_q_table_display():
    row = 0
    for state, actions in q_table.items():
        state_label = tk.Label(frame, text=f"State {state}", font=header_font, bg=header_color, borderwidth=1,
                               relief="solid")
        state_label.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")

        col = 1
        for action, q_value in actions.items():
            action_label = tk.Label(frame, text=f"{action}: {q_value:.2f}", font=q_value_font, bg="white",
                                    borderwidth=1, relief="solid")
            action_label.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            col += 1

        row += 1


# Create the Q-table display
create_q_table_display()

# Adjust column weights to expand the grid cells
for col in range(5):
    frame.grid_columnconfigure(col, weight=1)

# Run the Tkinter main loop
root.mainloop()
