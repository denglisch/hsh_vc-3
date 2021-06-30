# Visual Computing: Wavelets for Computer Graphics
# team01

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
import numpy as np


def render_points_array(ax, control_points_array=None, points_array=None, discrete=False):
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

    if control_points_array is not None:
        for i, cp in enumerate(control_points_array):
            disc = "c{}".format(i)
            #ax.add_patch(plt.Circle(cp, radius=r_cp, fc=c_cp, fill=True))
            #ax.annotate(disc, cp + ann_offset)

            if i > 0:
                from_p = control_points_array[i - 1]
                to_p = cp
                ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=0.0, ls='--', lw=1.0, color=c_cc)

    if points_array is not None:
        for i,p in enumerate(points_array):
            if discrete:
                ax.add_patch(plt.Circle(p, radius=r_p, fc=c_p, fill=False))
            if i>0:
                from_p=points_array[i-1]
                to_p=p
                ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=0.0, ls='-', lw=1.0, color=c_sc)
    return

def build_plt(original_curve_points_array, get_new_points_to_draw_for_level,
              update_point_in_pointlist,
              set_pumping_method_by_doubling,
              slider_max_level=10):
    #prepare plot
    fig, ax = plt.subplots(figsize=[12, 12])
    #init vis
    level=slider_max_level

    #render_points_array(ax, curve_points_array, points_array, level)
    #Slider (widget example adapted from: https://riptutorial.com/matplotlib/example/23577/interactive-controls-with-matplotlib-widgets)
    #slider axes
    slider_ax = plt.axes([0.35, .03, 0.50, 0.02])
    level_slider = Slider(slider_ax, "Level", 1.0, slider_max_level, valinit=level, valstep=0.1)

    points_array=original_curve_points_array
    discrete=False

    #defined locally to have all values here
    def update_vis(val):
        #get selected step
        nonlocal level,points_array

        level=val

        points_array=get_new_points_to_draw_for_level(level)
        #print(points_array)

        #print("Replot at Level {} on {} control points with {} total points".format(step, count_cp, count_p))

        #rebuild vis on axes
        render_points_array(ax, original_curve_points_array, points_array, discrete)
        fig.canvas.draw_idle()
    update_vis(level)
    # call update function on slider value change
    level_slider.on_changed(update_vis)

    radio_ax = plt.axes([0.01, 0.13, 0.1, 0.1])  # , facecolor=axcolor)
    radio = RadioButtons(radio_ax, ('discrete', 'fractional-\nlevel'),1)

    def set_discrete(label):
        nonlocal discrete
        nonlocal level_slider
        if label=='discrete':
            discrete=True
            level_slider.set_val(round(level_slider.val))
            level_slider.valstep=1.0
        else:
            discrete=False
            level_slider.valstep=0.1
            update_vis(level)
    radio.on_clicked(set_discrete)

    radio_ax2 = plt.axes([0.01, 0.03, 0.1, 0.1])
    radio2 = RadioButtons(radio_ax2, ('frac. by\ndoubling', 'frac. by\naveraging'),1)
    def set_discrete(label):
        if label=='frac. by\ndoubling':
            set_pumping_method_by_doubling(True)
        else:
            set_pumping_method_by_doubling(False)
        if not discrete:
            update_vis(level)
    radio2.on_clicked(set_discrete)

    def on_key_press(event):
        #print('press', event.key)
        if event.key == None:
            return
        nonlocal level_slider
        if event.key == 'left':
            if level_slider.val>1.0:
                level_slider.set_val(level_slider.val-level_slider.valstep)
        if event.key == 'right':
            if level_slider.val<slider_max_level:
                level_slider.set_val(level_slider.val+level_slider.valstep)
        #fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # this was to define points for initial curve ;)
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print("[{}, {}],".format(ix,iy))
    #fig.canvas.mpl_connect('button_press_event', onclick)

    #from: https://stackoverflow.com/questions/50439506/dragging-points-in-matplotlib-interactive-plot
    pind = None  # active point
    epsilon = 5  # max pixel distance
    def button_press_callback(event):
        'whenever a mouse button is pressed'
        if not discrete:
            return
        nonlocal pind
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        # print(pind)
        pind = get_ind_under_point(event)
    def button_release_callback(event):
        'whenever a mouse button is released'
        nonlocal pind
        if event.button != 1:
            return
        pind = None
    def get_ind_under_point(event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        # print('display x is: {0}; display y is: {1}'.format(event.x,event.y))
        t = ax.transData.inverted()
        tinv = ax.transData
        xy = t.transform([event.x, event.y])
        #print('data x is: {0}; data y is: {1}'.format(xy[0],xy[1]))
        xr = np.reshape(points_array[:,0], (np.shape(points_array[:,0])[0], 1))
        yr = np.reshape(points_array[:,1], (np.shape(points_array[:,1])[0], 1))
        xy_vals = np.append(xr, yr, 1)
        xyt = tinv.transform(xy_vals)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        # print(d[ind])
        if d[ind] >= epsilon:
            ind = None
        #print(ind)
        return ind

    def motion_notify_callback(event):
        'on mouse movement'
        if not discrete:
            return
        nonlocal points_array
        if pind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        # update yvals
        # print('motion x: {0}; y: {1}'.format(event.xdata,event.ydata))
        points_array[pind] = [event.xdata,event.ydata]
        update_point_in_pointlist(pind, [event.xdata,event.ydata])

        # update curve via sliders and draw
        #update_vis(level)
        render_points_array(ax, original_curve_points_array, points_array, discrete)
        #sliders[pind].set_val(yvals[pind])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', button_press_callback)
    fig.canvas.mpl_connect('button_release_event', button_release_callback)
    fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)

    #finally show plot
    plt.show()
    return