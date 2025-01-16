# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make the schemdraw diagram of how to connect the buttons, sensors, mux, and Arduino
to the computer.
"""

__author__ = "Kyle Vitautas Lopin"

import matplotlib.pyplot as plt
import matplotlib.patches as patches

FONTSIZE = 9
# Set font to 'Arial' 'Times' or 'Symbol'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': FONTSIZE})  # Set consistent font size
WIDTH = 0.8
LEFT = 0.1

class BreakoutBoard:
    def __init__(self, x, y, width=0.25, height=0.1, color="lightblue", label=None):
        """
        Initialize a BreakoutBoard object.

        Parameters:
        x (float): X position of the board's bottom-left corner.
        y (float): Y position of the board's bottom-left corner.
        width (float): Width of the breakout board (default: 0.2).
        height (float): Height of the breakout board (default: 0.1).
        color (str): Color of the breakout board (default: "lightblue").
        label (str): Optional label for the breakout board.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.label = label

    def draw(self, ax):
        """Draw the breakout board on the provided axis."""
        # Draw the rectangle (breakout board)
        rect = patches.FancyBboxPatch((self.x, self.y), self.width, self.height, linewidth=2,
                                 edgecolor="black", facecolor=self.color,
                                 boxstyle="round,pad=0.05")
        ax.add_patch(rect)

        # If a label is provided, place it inside the rectangle
        if self.label:
            ax.text(self.x + self.width / 2, self.y + self.height / 2, self.label, color="black",
                    ha="center", va="center")


class BlockDiagram:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(3, 4))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-0.5, 1)
        # Remove the box while keeping ticks and labels
        for spine in ['top', 'right', 'left', 'bottom']:
            self.ax.spines[spine].set_visible(False)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def add_breakout_board(self, x, y, width=0.1, height=0.1, color="lightblue", label=None):
        board = BreakoutBoard(x, y, width, height, color, label)
        board.draw(self.ax)

    def draw_line(self, x_start, y_start, x_end, y_end):
        """Draw an arrow between two points."""
        self.ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start))


# Create the diagram
diagram = BlockDiagram()

# Add Sensor \ Button pairs
x_positions = [0.15, 0.45, 0.75]
sensors = ["AS7262", "AS7263", "AS7265x"]
for x_pos, sensor in zip(x_positions, sensors):
    diagram.add_breakout_board(x_pos, 0.5, width=0.13,
                               label="Button", color="lightgreen")
    # Lines from buttons to sensors
    plt.plot([x_pos + 0.05, x_pos + 0.05], [0.65, 0.75], 'k', lw=2)
    diagram.add_breakout_board(x_pos, 0.8, width=0.13,
                               label=sensor, color="indianred")
    # These are the lines for sensor to mux
    plt.plot([x_pos + 0.05, x_pos + 0.05], [0.45, 0.35], 'k', lw=2)

diagram.add_breakout_board(LEFT, 0.2, width=WIDTH,
                           label="Qwiic Mux", color="indianred")
# add line between Arduino and Mux
plt.plot([0.5, 0.5], [0.15, 0.05], 'k', lw=2)
diagram.add_breakout_board(LEFT, -0.1, width=WIDTH, label="Qwiic Arduino")
plt.plot([0.5, 0.5], [-0.25, -0.15], 'k', lw=2)
diagram.add_breakout_board(LEFT, -0.4, width=WIDTH,
                           label="Computer", color="lightgray")

# Show the diagram
plt.tight_layout()
# plt.show()

plt.savefig("diagram_setup.jpeg", dpi=600)
