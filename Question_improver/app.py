"""First easy implementation of the recommendation system."""
import warnings

import tkinter as tk

from editor.get_recomendations import get_recommendation_report

warnings.filterwarnings('ignore')

# variable to switch between key triggered and time triggered evaluation
KEY_RELEASE_TRIGGERED = True
time_trigger_id = None


def select_trigger():
    """Switch used evaluation trigger between key release and time based.
    """
    global KEY_RELEASE_TRIGGERED, time_trigger_id
    # if currently key triggered, unbind
    if KEY_RELEASE_TRIGGERED:
        text.unbind('<KeyRelease>')
    # if currently time triggered, turn off
    else:
        root.after_cancel(time_trigger_id)
    # negate trigger
    KEY_RELEASE_TRIGGERED = not KEY_RELEASE_TRIGGERED
    # turn on key release trigger
    if KEY_RELEASE_TRIGGERED:
        curr_trig.config(text='key released trigger')
        text.bind('<KeyRelease>', evaluate_question)
    # turn on the time trigger
    else:
        curr_trig.config(text='evaluation triggered every two seconds')
        time_trigger_id = root.after(2000, evaluate_question)


def evaluate_question(event=None):
    """Suggest recommendation to the user input question.
    """
    user_input = text.get("1.0", tk.END)[:-1]
    if user_input:
        report.config(text=get_recommendation_report(user_input))
    if not KEY_RELEASE_TRIGGERED:
        time_trigger_id = root.after(2000, evaluate_question)


root = tk.Tk()
root.title('Question enhancement')

label = tk.Label(root, text='What is your question?')
label.pack(pady=20, padx=20)

button = tk.Button(
    root,
    text='switch evaluation trigger',
    command=select_trigger
)
button.pack(pady=5)

curr_trig = tk.Label(root, text='key released trigger')
curr_trig.pack(pady=5)

text = tk.Text(root, wrap=tk.WORD)
text.pack(pady=20)
# initial evaluation binding
text.bind('<KeyRelease>', evaluate_question)

report = tk.Label(root, text='Report will show here.', justify='left')
report.pack(pady=20, padx=20)

root.mainloop()
