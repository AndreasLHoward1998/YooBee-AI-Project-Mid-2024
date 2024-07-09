import pandas as pd
import tkinter as tk
from tkinter import ttk

# Load the CSV file
df = pd.read_csv('EEG_Analysis.csv')import pandas as pd
import tkinter as tk
from tkinter import ttk
from docx import Document

# This is to load the CSV file
df = pd.read_csv('EEG_Analysis.csv')

# Next is to load the Word document
doc = Document('Intro to Machine Learning - activity.docx')

# Then is to extract text from the Word document
doc_text = '\n'.join([para.text for para in doc.paragraphs])

# And now, we print out the column names and first few rows for debugging
print("Column Names:", df.columns)
print(df.head())

# This will create a Tkinter window
root = tk.Tk()
root.title("EEG Analysis and Document Display")

# Then we create a Text widget to display the document text
text_widget = tk.Text(root, wrap='word')
text_widget.pack(expand=True, fill='both')

# And finally we insert the document text into the Text widget
text_widget.insert('1.0', doc_text)

# This function is to map values to colors
def value_to_color(value, value_range, color_range):
    """ Map a value to a color in a given range. """
    min_val, max_val = value_range
    min_color, max_color = color_range
    
    # This is to normalize the value to be within 0 and 1
    norm_value = (value - min_val) / (max_val - min_val)
    
    # This will interpolate the color
    r = int(min_color[0] + (max_color[0] - min_color[0]) * norm_value)
    g = int(min_color[1] + (max_color[1] - min_color[1]) * norm_value)
    b = int(min_color[2] + (max_color[2] - min_color[2]) * norm_value)
    
    return f'#{r:02x}{g:02x}{b:02x}'

# This is the function to update text color based on engagement values
def update_text_colors():
    for i, row in df.iterrows():
        engagement = row['Engagement']
        memory_commitment = row['Memory Commitment']
        
        # This will define the color ranges
        engagement_color = value_to_color(engagement, (0, 1), ((255, 0, 0), (0, 0, 255))) # Red to Blue
        memory_commitment_color = value_to_color(memory_commitment, (0, 1), ((0, 255, 0), (255, 255, 0))) # Green to Yellow
        
        text_widget.tag_add(f'engagement_{i}', f'{i + 1}.0', f'{i + 1}.end')
        text_widget.tag_configure(f'engagement_{i}', foreground=engagement_color, background=memory_commitment_color)

update_text_colors()

root.mainloop()


# Print out the column names and first few rows
print("Column Names:", df.columns)
print(df.head())

# Create a Tkinter window
root = tk.Tk()
root.title("EEG Analysis")

# Create a Treeview widget
tree = ttk.Treeview(root)
tree['columns'] = list(df.columns)
tree.heading('#0', text='Index', anchor='w')
for col in df.columns:
    tree.heading(col, text=col, anchor='w')

# Insert data into the Treeview
for index, row in df.iterrows():
    tree.insert('', 'end', text=index, values=list(row))

tree.pack(expand=True, fill='both')

# Function to update colors based on engagement and memory commitment
def update_colors():
    for item in tree.get_children():
        values = tree.item(item, 'values')
        engagement = float(values[df.columns.get_loc('Engagement')])
        memory_commitment = float(values[df.columns.get_loc('Memory Commitment')])
        if engagement > 0.1:
            color = 'blue'
        else:
            color = 'red'
        tree.item(item, tags=(color,))
    tree.tag_configure('red', background='red', foreground='white')
    tree.tag_configure('blue', background='blue', foreground='white')

update_colors()

root.mainloop()
