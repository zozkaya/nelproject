import tkinter as tk
import time

def create_bci_gui(gesture):
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

    display_gestures(window, gesture)

    # Start the GUI event loop
    window.after(1000, lambda: update_labels(window, gesture))
    window.mainloop()

def update_labels(window, gesture):
    # Replace this with your actual data update logic
    current_time = time.strftime("%H:%M:%S")

    # Update the detected gesture label
    subtitle_label.config(text="Detected gesture: " + current_time)

    # Update the arm movement label
    subtitle_label2.config(text="Arm movement: " + current_time)

    # Schedule the next update
    window.after(1000, lambda: update_labels(window, gesture))

def display_gestures(window, gesture):
    # Your existing display_gestures function remains unchanged

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

    global subtitle_label, subtitle_label2
    subtitle_label = tk.Label(window, text="Detected gesture: " + human_gesture, font=("Helvetica", 15), fg='black')
    subtitle_label.pack(pady=5)
    subtitle_label.configure(bg="white")

    subtitle_label2 = tk.Label(window, text="Arm movement: " + gesture, font=("Helvetica", 15), fg='black')
    subtitle_label2.pack(pady=5)
    subtitle_label2.configure(bg="white")
