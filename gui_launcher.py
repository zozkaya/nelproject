import tkinter as tk
import time
from classify_gesture import classify_gesture


def create_bci_gui():
    # Create the main window
    window = tk.Tk()
    window.title("BCI GUI")

    # Set the size of the window
    window.geometry("600x500")  # Width x Height

    # Set the background color to white
    window.configure(bg="white")

    # Create and place widgets
    title_label = tk.Label(window, text="EMG Controlled Robotic Arm", font=("Helvetica", 20, "bold"), fg='black')
    title_label.pack(pady=10)
    title_label.configure(bg="white")


    display_gestures(window,gesture)

    # Start the GUI event loop
    window.mainloop()

def display_gestures(window,gesture):
     
    if gesture == "no movement":
        human_gesture = "resting state"
    elif gesture == "clamp close":
        human_gesture = "hand to fist"
    elif gesture == "clamp open":
        human_gesture = "hand open"
    elif gesture == "wrist left":
        human_gesture = "wrist left"
    elif gesture == "wrist right":
        human_gesture = "wrist right"
    elif gesture == "arm out":
        human_gesture = "two-finger pinch"
    elif gesture == "arm in":
        human_gesture = "index finger point"
    elif gesture == "arm up":
        human_gesture = "wrist up"
    elif gesture == "arm down":
        human_gesture = "wrist down"

    subtitle_label = tk.Label(window, text="Detected gesture: " + human_gesture, font=("Helvetica", 15), fg='black')
    subtitle_label.pack(pady=5)
    subtitle_label.configure(bg="white")

    subtitle_label2 = tk.Label(window, text="Arm movement: " + gesture, font=("Helvetica", 15), fg='black')
    subtitle_label2.pack(pady=5)
    subtitle_label2.configure(bg="white")


# Call the function to create the GUI
create_bci_gui()