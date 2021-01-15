import numpy as np
import math

import h5py

import matplotlib.pyplot as plt
import matplotlib.colors as colors

def make_similarity_analysis_plot(chosen_test_examples, folder_influence, folder_model, save = False, influence_figure_name = 'Fig8_similarity_analysis', format = 'pdf'):

    training_mask = np.load('influence_functions/data_and_masks/phase_diagram_training_mask.npy') 
    test_mask = np.load('influence_functions/data_and_masks/phase_diagram_test_mask.npy')
    data = h5py.File('./data/phase_diagram_rephased.h5', 'r') #
    test_data = data

    freq = np.array(data['parameter/freq']) # from 5.1 to 7.8 kHz, every 0.1
    phase = np.array(data['parameter/phase']) # from -180 to 180, every 5

    test_freq = np.array(freq[test_mask])
    test_phase = np.array(phase[test_mask])
    training_freq = np.array(freq[training_mask])
    training_phase = np.array(phase[training_mask])

    # Theory data
    file = h5py.File('./data/phase_diagram_theory.h5','r')
    file.keys()

    chern_number = file["chern_number"][:]
    freq0 = file["freq"][:]
    phase0 = file["phase"][:]
    x0,y0 = np.meshgrid(phase0,freq0)

    # Unique phases and freqs
    uphase = np.unique(phase)
    ufreq = np.unique(freq)
    X,Y = np.meshgrid(uphase,ufreq)

    def create_out(test_sample, f = np.mean):
        with open(folder_influence + '/exact_influence_test' + str(test_sample) + '.txt') as filelabels:
            influence_functions = np.loadtxt(filelabels, dtype=float)

            uphase = np.unique(phase)
            ufreq = np.unique(freq)

            out = np.zeros((len(uphase),len(ufreq)))
            dout = np.zeros((len(uphase),len(ufreq)))

            for i,ph in enumerate(uphase):
                for j,fr in enumerate(ufreq):
                    mask = np.intersect1d(np.where(training_freq == fr),np.where(training_phase == ph))
                    out[i,j] = f(influence_functions[mask])
                    dout[i,j] = np.std(influence_functions[mask])

        return out, dout, influence_functions

    # for consistency with co-authors
    face_colors = {
        'orange': [0.8906, 0.4609, 0.4062],
        'gray': [0.6523, 0.6484, 0.6484],
        'blue': [0.5156, 0.5977, 0.8789]}
    edge_colors = {
        'orange': [0.9961, 0, 0],
        'gray': [0.4805, 0.4766, 0.4766],
        'blue': [0, 0, 0.9961]}

    edge_colors_arr = [ec for ec in edge_colors.values()]
    face_colors_arr = [fc for fc in face_colors.values()]

    cycler = (plt.cycler(mec=edge_colors_arr)+
            plt.cycler(mfc=face_colors_arr)+
            plt.cycler(color=edge_colors_arr))

    plt.rcParams.update({
        #'figure.figsize': (12, 8),
        'font.size': 12,
        #'lines.markeredgewidth': 2,
        #'lines.markersize': 9,
        #'lines.marker': 'x',
        #'lines.linestyle': '-',
        #'lines.linewidth': 3,
        'axes.prop_cycle': cycler
        })

    ## ALL TOGETHER ##
    outs, douts, inf_funs = [None] * 3, [None] * 3, [None] * 3
    for i in range(3):
        outs[i], douts[i], inf_funs[i] = create_out(chosen_test_examples[i], f=np.mean)

    fig, axs = plt.subplots(2, 3, gridspec_kw={'height_ratios': [3, 2.2]}, figsize=(14,6), sharey="row")
    fillalpha, fillcolor = 0.5, "c"
    cmap = "jet"
    vmin, vmax = np.nanmin(outs), np.nanmax(outs)

    #lintresh_divider = 120
    #symlog_treshold = vmax/lintresh_divider
    symlog_treshold = 1e-6

    for i in range(3):
        test_sample = chosen_test_examples[i]

        # Test data.
        X_test = test_phase[test_sample]
        Y_test = test_freq[test_sample]

        pick_freq = np.round((Y_test - 5.1) / 0.1).astype(int)

        ax = axs[0,i]
        im = ax.pcolormesh(X,Y,outs[i].T,cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, rasterized=True, norm=colors.SymLogNorm(linthresh=symlog_treshold, vmin=vmin, vmax=vmax))
        im.set_edgecolor('face')
        #fig.colorbar(im,ax=ax)
        ax.contour(x0,y0,np.round(chern_number.T),levels=2)
        ax.scatter(X_test, Y_test, c='black', marker="X", s=80)
        dx,dy = 0,0
        ax.set_xlim(uphase.min()-dx,uphase.max()+dx)
        ax.set_ylim(ufreq.min(),ufreq.max())
        ax.set_xticks([-180,-90,0,90,180])
        ax.plot([uphase.min(), uphase.max()], [Y_test, Y_test], "--", c='black')
        
        ax = axs[1,i]
        label = None
        if i == 0:
            label = "data"
            label1 = "training region"
        ax.errorbar(X[pick_freq],outs[i].T[pick_freq],yerr=douts[i].T[pick_freq],fmt=".",label=label)
        ax.text(0.995, 0.04, "$f_{SH}$" + f" = {Y_test} kHz", transform = ax.transAxes, horizontalalignment = 'right') #, bbox=propshorizontalalignment='center', verticalalignment='center',
        ax.set_xticks([-180,-90,0,90,180])
        ax.set_xlim(uphase.min()-dx,uphase.max()+dx)
        ax.set_yscale('symlog', linthreshy=symlog_treshold)
        #ax.axvline(0, color='black', label="theoretical transition")

    # Fits
    fitcolor = edge_colors["gray"]

    axs[0,0].set_ylabel("Shaking\n Frequency, $f_{SH}$ (kHz)",fontsize=16)
    #axs[1,0].set_ylabel("Inf. fun. value",fontsize=16)
    axs[1,0].set_ylabel("$\mathcal{I}$",fontsize=16)
    letters = ['a', 'b', 'c']
    for i in range(3):
        #axs[0,i].set_xlabel("Shaking Phase (°)",fontsize=16)
        axs[1,i].set_xlabel("Shaking Phase (°)",fontsize=16)
        axs[1,i].grid()
        axs[0,i].text(-180, 7.9, letters[i], fontsize=16, fontweight='bold') #, transform = axs[0,0].transAxes

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.5, 0.02, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_title("$\mathcal{I}$")

    plt.show()

    if save is True:
        plt.savefig("./figures/" + influence_figure_name + '.' + format, dpi=450)

def make_micromotion_removal_plot(test_sample, folder_influence_before, folder_influence_after, save = False, influence_figure_name = 'Fig4_micromotion_removal', format = 'pdf'):
    phase_ticks = np.arange(-180, 181, 90)
    freq_limit_low = 5.1
    freq_limit_high = 7.8

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    original_cmap = plt.get_cmap('jet')
    cmap = truncate_colormap(original_cmap, 0, 0.98)

    face_colors = {
        'orange': [0.8906, 0.4609, 0.4062],
        'gray': [0.6523, 0.6484, 0.6484],
        'blue': [0.5156, 0.5977, 0.8789]
    }

    edge_colors = {
        'orange': [0.9961, 0, 0],
        'gray': [0.4805, 0.4766, 0.4766],
        'blue': [0, 0, 0.9961]
    }

    edge_colors_arr = [ec for ec in edge_colors.values()]
    face_colors_arr = [fc for fc in face_colors.values()]

    cycler = (plt.cycler(mec=edge_colors_arr)+
            plt.cycler(mfc=face_colors_arr)+
            plt.cycler(color=edge_colors_arr))

    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 14,
        'lines.markeredgewidth': 2,
        'lines.markersize': 9,
        'lines.marker': 'o',
        'lines.linestyle': '',
        'lines.linewidth': 3,
        'axes.prop_cycle': cycler
        })

    # Panel a
    with h5py.File('./data/phase_diagram_theory.h5', 'r') as theory_file:
        theory_chern_numbers = theory_file['chern_number'][:] # note that we use [:] to copy the data into memory
        theory_freqs = theory_file['freq'][:]
        theory_phase = theory_file['phase'][:]

    # for plotting we transpose the chern numbers to have the phase as x axis.
    theory_chern_numbers = theory_chern_numbers.T

    # make axis grid
    [freq_axis, phase_axis] = np.meshgrid(theory_phase, theory_freqs)

    ## Fig.
    fig, axs = plt.subplots(1, 3, figsize=(14,4), sharey="row")
    letters = ['a', 'b', 'c']

    for i in range(3):
        axs[i].grid()
        axs[i].set_ylim(freq_limit_low, freq_limit_high)
        if i == 0:
            axs[i].text(-180, 7.9, letters[i], fontsize=16, fontweight='bold') #, transform = axs[0,0].transAxes
        else:
            axs[i].text(0, 7.9, letters[i], fontsize=16, fontweight='bold') #, transform = axs[0,0].transAxes

    axs[0].contour(freq_axis, phase_axis, theory_chern_numbers, levels=(-0.5, 0.5), cmap='seismic')
    axs[0].plot([-90, -90], [freq_limit_low, freq_limit_high], "--", c='black')
    axs[0].set_xticks(phase_ticks)
    axs[0].set_xlim(-180, 180)
    axs[0].set_xlabel('Shaking Phase (°)')
    axs[0].set_ylabel('Shaking Frequency (kHz)')
    #plt.savefig('plots/01_theory_phase_diagram.eps')

    for i in [1,2]:

        axs[i].set_axisbelow(True)
        
        if i == 1:
            folder_influence = folder_influence_before
            data = h5py.File('./data/validation_single_cut_56.h5', 'r')
            test_data = h5py.File('./data/single_cut_56.h5', 'r')
            training_mask = np.load('influence_functions/data_and_masks/validation_single_cut_training_mask.npy') 
            test_mask = np.load('influence_functions/data_and_masks/single_cut_test_mask.npy')

        else:
            folder_influence = folder_influence_after
            data = h5py.File('./data/validation_single_cut_rephased.h5', 'r')
            test_data = h5py.File('./data/single_cut_rephased.h5', 'r')
            training_mask = np.load('influence_functions/data_and_masks/validation_single_cut_training_mask.npy') 
            test_mask = np.load('influence_functions/data_and_masks/single_cut_test_mask.npy')

        freq = np.array(data['parameter/freq'])
        micromotion = np.array(data['parameter/micromotion_phase'])
        micromotion = micromotion / np.pi

        test_freq = np.array(test_data['parameter/freq'])
        test_micromotion = np.array(test_data['parameter/micromotion_phase'])
        test_micromotion = test_micromotion / np.pi

        Y = freq[training_mask]
        X = micromotion[training_mask]

        # Influence functions of all train elements for one test example
        with open(folder_influence + '/exact_influence_test' + str(test_sample) + '.txt') as filelabels:
            influence_functions = np.loadtxt(filelabels, dtype=float)

            # Make test data.
            Y_test = test_freq[test_mask][test_sample]
            X_test = test_micromotion[test_mask][test_sample]

            sorting_indices = np.argsort(influence_functions)

            # V_min and v_max
            vmax = np.amax(influence_functions)
            vmin = np.amin(influence_functions)

            lintresh_divider = 10
            symlog_treshold = vmax/lintresh_divider

            # Chosen test point for publication, to ensure both subplots have the same color map
            if test_sample == 14:
                vmax = 0.0525433496
                vmin = -0.0541656874
                symlog_treshold = 1e-2

            size = 80

            # Data for three-dimensional scattered points
            lintresh_divider = 10
            surf = axs[i].scatter(X, Y, c=influence_functions, cmap=cmap, vmin=vmin, vmax=vmax, norm=colors.SymLogNorm(linthresh=symlog_treshold, vmin=vmin, vmax=vmax))
            axs[i].scatter(X[sorting_indices[:5]], Y[sorting_indices[:5]], c=influence_functions[sorting_indices[:5]], cmap=cmap, vmin=vmin, vmax=vmax, marker="D", s=size, norm=colors.SymLogNorm(linthresh=symlog_treshold, vmin=vmin, vmax=vmax))
            axs[i].scatter(X[sorting_indices[-5:]], Y[sorting_indices[-5:]], c=influence_functions[sorting_indices[-5:]], cmap=cmap, vmin=vmin, vmax=vmax, marker="D", s=size, norm=colors.SymLogNorm(linthresh=symlog_treshold, vmin=vmin, vmax=vmax))
            axs[i].scatter(X_test, Y_test, c='black', marker="X", s=size)

            axs[i].set_xlabel('Micromotion Phase ($\pi$)')
            axs[i].set_xlim(0,2)

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.4])
    fig.colorbar(surf, cax=cbar_ax)
    cbar_ax.set_title("$\mathcal{I}$")

    if save is True:
        plt.savefig("./figures/" + influence_figure_name + '.' + format, dpi=450)
    else:
        plt.show()
