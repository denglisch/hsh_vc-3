import Haar
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# TODO:
# plt with slider for iteration steps
# one fig is rendered (control points and curve)
# subdivision

#TODO:
# task 4 and 5

def main():
    #control_points_array = np.random.rand(10, 2)
    control_points_array = [[0.12043010752688174, 0.7251082251082251],
        [0.25806451612903225, 0.8777056277056279],
        [0.42258064516129035, 0.5876623376623378],
        [0.6709677419354839, 0.5822510822510824],
        [0.5591397849462366, 0.8614718614718615],
        [0.8849462365591397, 0.8354978354978355],
        [0.7935483870967742, 0.7424242424242425],
        [0.9419354838709678, 0.48917748917748927],
        [0.7516129032258064, 0.06709956709956713],
        [0.48602150537634414, 0.22077922077922082],
        [0.1870967741935484, 0.04112554112554115]]
    print(control_points_array)
    points_array=[]
    #print(points_array)
    render(control_points_array, points_array)
    return

def render_subdivision_step_update(control_points_array, points_array, step, ax):
    ax.clear()
    ax.axis([0.0, 1.0, 0.0, 1.0])

    #radius
    r_cp=0.0025
    r_p=0.0025
    #color
    c_cp='black'
    c_p='blue'
    c_sc='red'
    c_cc='gray'

    ann_offset = np.array([0.006, 0.003])  # offset for annotations

    legend: list = []

    if control_points_array is not None:
        for i, cp in enumerate(control_points_array):
            disc = "c{}".format(i)
            ax.add_patch(plt.Circle(cp, radius=r_cp, fc=c_cp, fill=True))
            ax.annotate(disc, cp + ann_offset)

            if i == 0:
                from_p = control_points_array[len(control_points_array) - 1]
            else:
                from_p = control_points_array[i - 1]
            to_p = cp
            ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=2.0, ls='-', lw=1.0, color=c_cc)

        #disc="Control Points"
        #if disc not in legend:
        #legend.append(disc)

    if points_array is not None:
        for i,p in enumerate(points_array):
            ax.add_patch(plt.Circle(p, radius=r_p, fc=c_p, fill=False))
            #if "Points" not in legend: legend.append("Points")

            if i==0:
                from_p=points_array[len(points_array)-1]
            else:
                from_p=points_array[i-1]
            to_p=p
            ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=2.0, ls='-', lw=1.0, color=c_sc)
            #if "Curve" not in legend: legend.append("Curve")

    #ax.legend(legend, loc="lower right")
    return

def render(control_points_array, points_array):
    # controlpoints as dots
    # points as circles
    # curve as line (acc. 2.2)

    #def visualize_device_2d(meas, beacons, time):
    """
    Builds a mathplotlib widget with slider for timestamps.
    Calls visualize_device_in_time_update()
    """

    #prepare plot
    fig, ax = plt.subplots(figsize=[12, 12])
    #init vis
    render_subdivision_step_update(control_points_array, points_array, 0, ax)
    #Slider (widget example adapted from: https://riptutorial.com/matplotlib/example/23577/interactive-controls-with-matplotlib-widgets)
    #slider axes
    slider_ax = plt.axes([0.35, .03, 0.50, 0.02])
    subdivision_slider = Slider(slider_ax, "Subdivision Step", 0, 10, valinit=0, valstep=1)
    #defined locally to have all values here
    def update_vis(step):
        #get selected step

        #TODO: recalc points_array
        points_array=calc_subdivision(control_points_array, step)

        print("Replot Subdivision Step {}".format(step))
        #rebuild vis on axes
        render_subdivision_step_update(control_points_array, points_array, step, ax)
        fig.canvas.draw_idle()

    # call update function on slider value change
    subdivision_slider.on_changed(update_vis)

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print("[{}, {}],".format(ix,iy))
    fig.canvas.mpl_connect('button_press_event', onclick)

    #finally show plot
    plt.show()
    return

def subdivision_function(points_array):
    for i, p in enumerate(points_array):
        if i%2==1:
            previous=points_array[i - 1] if i>0 else points_array[len(points_array)-1]
            next = points_array[i + 1] if i<len(points_array)-1 else points_array[0]
            points_array[i]=0.5*(previous+next)
    return points_array

def calc_subdivision(control_points_array, steps, subdivision_function=subdivision_function):
    new_points_array=control_points_array
    for i in range(0,steps+1):
        #split #each point is now doubled
        points_array=np.repeat(new_points_array, 2, axis=0)

        #avg step
        new_points_array=subdivision_function(points_array)

    return new_points_array

if __name__ == "__main__":
    # execute only if run as a script
    main()
