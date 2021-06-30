# Visual Computing: Wavelets for Computer Graphics
# team01

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
import numpy as np
import math

def main():
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

    calc_and_render_subdivision(control_points_array)
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

def calc_and_render_subdivision(control_points_array):
    # controlpoints as dots
    # points as circles
    # curve as line (acc. 2.2)

    #prepare plot
    fig, ax = plt.subplots(figsize=[12, 12])
    #init vis
    cur_step=0
    points_array = calc_subdivision(control_points_array, cur_step)
    render_subdivision_step_update(control_points_array, points_array, cur_step, ax)
    #Slider (widget example adapted from: https://riptutorial.com/matplotlib/example/23577/interactive-controls-with-matplotlib-widgets)
    #slider axes
    slider_ax = plt.axes([0.35, .03, 0.50, 0.02])
    subdivision_slider = Slider(slider_ax, "Subdivision Step", 0, 10, valinit=cur_step, valstep=1)

    subdivision_scheme_function=subdivision_chaikin

    #defined locally to have all values here
    def update_vis(step):
        #get selected step
        nonlocal cur_step
        cur_step=step

        points_array=calc_subdivision(control_points_array, step, subdivision_scheme_function=subdivision_scheme_function)

        count_cp=len(control_points_array)
        count_p=count_cp*math.pow(2,step)#+step
        print("Replot Subdivision Step {} on {} control points with {} total points".format(step, count_cp, count_p))

        #rebuild vis on axes
        render_subdivision_step_update(control_points_array, points_array, step, ax)
        fig.canvas.draw_idle()

    # call update function on slider value change
    subdivision_slider.on_changed(update_vis)

    # from: https://matplotlib.org/2.0.2/examples/widgets/radio_buttons.html
    #axcolor = 'lightgoldenrodyellow'
    radio_ax = plt.axes([0.01, 0.03, 0.15, 0.15])#, facecolor=axcolor)
    radio = RadioButtons(radio_ax, ('Chaikin', 'Daubechie', 'DLG', 'B-Spline cubic'))
    def select_scheme(label):
        hzdict = {'Chaikin': subdivision_chaikin,
                  'Daubechie': subdivision_daubechie,
                  'B-Spline cubic': subdivision_b_spline_cubic,
                  'DLG': subdivision_dlg}
        nonlocal subdivision_scheme_function
        subdivision_scheme_function = hzdict[label]
        update_vis(cur_step)
    radio.on_clicked(select_scheme)


    # this was to define points for initial curve ;)
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print("[{}, {}],".format(ix,iy))
    fig.canvas.mpl_connect('button_press_event', onclick)

    #finally show plot
    plt.show()
    return

def splitting_step(points_array):
    points_array=np.repeat(points_array,2,axis=0)
    # each point is now doubled
    cp_points_array = np.copy(points_array)
    for i, p in enumerate(points_array):
        if i%2==1:
            next_p = points_array[i + 1] if i<len(points_array)-1 else points_array[0]
            cp_points_array[i]=0.5*(p+next_p)
    return cp_points_array

def subdivision_chaikin(points_array):
    cp_points_array = np.copy(points_array)
    for i, p in enumerate(points_array):
        previous_p = p
        next_p = points_array[i + 1] if i < len(points_array) - 1 else points_array[0]
        cp_points_array[i] = 0.5 * (previous_p + next_p)
    return cp_points_array

def subdivision_daubechie(points_array):
    cp_points_array = np.copy(points_array)
    one_plus_sqrt3=1.0+math.sqrt(3)
    one_minus_sqrt3=1.0-math.sqrt(3)
    for i, p in enumerate(points_array):
        previous_p = p
        next_p = points_array[i + 1] if i < len(points_array) - 1 else points_array[0]
        cp_points_array[i] = 0.5 * (one_plus_sqrt3*previous_p + one_minus_sqrt3*next_p)
    return cp_points_array

def subdivision_dlg(points_array):
    cp_points_array = np.copy(points_array)
    for i, p in enumerate(points_array):
        #keep points from prevoius step
        if i%2==1:
            prev_prev_p = points_array[i - 2] if i > 0 else points_array[len(points_array) - 2]
            prev_p = points_array[i - 1] if i > 0 else points_array[len(points_array) - 1]
            next_p = points_array[i + 1] if i < len(points_array) - 1 else points_array[0]
            next_next_p = points_array[i + 2] if i < len(points_array) - 1 else points_array[1]
            cp_points_array[i] = 1.0/16 * (-2.0*prev_prev_p+5.0*prev_p+10.0*p+5.0*next_p + -2.0*next_next_p)
    return cp_points_array

def subdivision_b_spline_cubic(points_array):
    cp_points_array = np.copy(points_array)
    for i, p in enumerate(points_array):
        previous_p = points_array[i - 1] if i > 0 else points_array[len(points_array) - 1]
        mid = p
        next_p = points_array[i + 1] if i < len(points_array) - 1 else points_array[0]
        cp_points_array[i] = 0.25 * (previous_p + 2*mid + next_p)
    return cp_points_array

def calc_subdivision(control_points_array, steps, subdivision_scheme_function=subdivision_dlg):
    points_array=control_points_array
    for i in range(0,steps):
        #SPLITTING STEP
        points_array=splitting_step(points_array)

        # AVG STEP
        points_array=subdivision_scheme_function(points_array)

    return points_array

if __name__ == "__main__":
    # execute only if run as a script
    main()
