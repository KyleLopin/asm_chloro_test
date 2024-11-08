# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
This will make the schemdraw diagram of how to connect the
"""

__author__ = "Kyle Vitautas Lopin"

import matplotlib.pyplot as plt
import matplotlib.patches as patches

FONTSIZE = 9

class BreakoutBoard:
    def __init__(self, x, y, width=0.2, height=0.1, color="lightblue", label=None):
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
                    ha="center", va="center", fontsize=FONTSIZE)


class BlockDiagram:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-0.5, 1)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def add_breakout_board(self, x, y, width=0.1, height=0.1, color="lightblue", label=None):
        board = BreakoutBoard(x, y, width, height, color, label)
        board.draw(self.ax)

    def draw_line(self, x_start, y_start, x_end, y_end):
        """Draw an arrow between two points."""
        self.ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start))

    def show(self):
        plt.show()


# Create the diagram
diagram = BlockDiagram()

# Add Sensor \ Button pairs
x_positions = [0.2, 0.45, 0.7]
sensors = ["AS7262", "AS7263", "AS7265x"]
for x_pos, sensor in zip(x_positions, sensors):
    diagram.add_breakout_board(x_pos, 0.8, label="Button", color="lightgreen")
    # Lines from buttons to sensors
    plt.plot([x_pos + 0.05, x_pos + 0.05], [0.65, 0.75], 'k', lw=2)
    diagram.add_breakout_board(x_pos, 0.5, label=sensor, color="indianred")
    # These are the lines for sensor to mux
    plt.plot([x_pos + 0.05, x_pos + 0.05], [0.45, 0.35], 'k', lw=2)

diagram.add_breakout_board(0.2, 0.2, width=0.6,
                           label="Qwiic Mux", color="indianred")
plt.plot([0.5, 0.5], [0.15, 0.05], 'k', lw=2)
diagram.add_breakout_board(0.2, -0.1, width=0.6, label="Qwiic Arduino")
plt.plot([0.5, 0.5], [-0.25, -0.15], 'k', lw=2)
diagram.add_breakout_board(0.2, -0.4, width=0.6,
                           label="Computer", color="lightgray")

# Show the diagram
# diagram.show()
plt.savefig("diagram_setup.svg")
