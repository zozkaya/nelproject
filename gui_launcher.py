import tkinter as tk
import time
from classify_gesture import classify_gesture

class RoboticArm(tk.Canvas):
    def __init__(self, master=None, **kwargs):
        tk.Canvas.__init__(self, master, **kwargs)
        self.base = self.create_rectangle(195, 250, 205, 400, fill="blue")  # Base of the arm
        self.arm = self.create_rectangle(195, 400, 205, 500, fill="red")  # Arm
        #self.bind("<Configure>", self.update_arm_position)

    def update_arm_position(self, event=None):
        self.coords(self.arm, self.winfo_width() / 2 - 5, self.winfo_height() / 2, self.winfo_width() / 2 + 5, self.winfo_height() / 2 + 100)

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

    # Create the robotic arm canvas
  #  robotic_arm_canvas = RoboticArm(window, width=400, height=400, bg="white")
    #robotic_arm_canvas.pack(pady=10)

    display_gestures(window)

    # Start the GUI event loop
    window.mainloop()

def display_gestures(window):
    gesture = classify_gesture()
    gesture = "arm up"

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
      #  move_robotic_arm_up(robotic_arm_canvas)
    elif gesture == "arm down":
        human_gesture = "wrist down"

    subtitle_label = tk.Label(window, text="Detected gesture: " + human_gesture, font=("Helvetica", 15), fg='black')
    subtitle_label.pack(pady=5)
    subtitle_label.configure(bg="white")

    subtitle_label2 = tk.Label(window, text="Arm movement: " + gesture, font=("Helvetica", 15), fg='black')
    subtitle_label2.pack(pady=5)
    subtitle_label2.configure(bg="white")

def move_robotic_arm_up(robotic_arm_canvas):
    for _ in range(50):
        robotic_arm_canvas.after(10, robotic_arm_canvas.update)
        robotic_arm_canvas.move(robotic_arm_canvas.arm, 0, -1)
        robotic_arm_canvas.after(10, time.sleep, 0.01)

def move_robotic_arm_down(robotic_arm_canvas):
    for _ in range(50):
        robotic_arm_canvas.after(10, robotic_arm_canvas.update)
        robotic_arm_canvas.move(robotic_arm_canvas.arm, 0, 1)
        robotic_arm_canvas.after(10, time.sleep, 0.01)

# Call the function to create the GUI
create_bci_gui()