# -*- coding: utf-8 -*-
"""
Plotting code for CDAC switching energy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tabulate import tabulate

# CDAC class located in cdac.py
from cdac import CDAC

def plot_total_energy(cdac):
    """
     Plot the total switching energy for each of the four different CDAC 
     circuits (conventional, two-step, charge sharing, and split cap).
     
     Parameters:
     -----------
     cdac : CDAC
         An object containing the energy consumption data for each circuit.
    
     Returns:
     --------
     None
     """
    plt.figure(1)
            
    plt.step(np.arange(code_num), cdac.energy_conv)
    plt.step(np.arange(code_num), cdac.energy_2step)
    plt.step(np.arange(code_num), cdac.energy_cs)
    plt.step(np.arange(code_num), cdac.energy_split)
    # plt.ylim(0, 30)
    
    # Set the labels for each curve
    plt.xlabel("Output Code")
    plt.ylabel("Switching Energy")
    plt.title("CDAC Energy Consumption")
    plt.legend(['Conventional', 'Two Step', 'Charge Sharing', 'Split Cap'])
    
    
    # Save as EPS file
    plt.savefig('energies.eps', format='eps', bbox_inches='tight')

def print_variances(cdac):
    """
    Calculates and prints the variances of the energy leakage for each capacitor switching method using data from a CDAC object.
    
    Parameters:
    -----------
    cdac : CDAC object
        A CDAC object containing the data to analyze.
    
    Returns:
    --------
    None
    """
    # fitting
    coefficients = np.polyfit(np.arange(code_num) - (code_num-1) / 2, cdac.energy_split, 4)
    coefficients[0] = 0
    coefficients[1] = 0
    coefficients[3] = 0
    # it fits the form ax^2 + b
    # split cap leaks in the form of distance squared from the middle code
    #plt.plot(np.polyval(coefficients, np.arange(code_num) - (code_num-1) / 2))
    '''
    '''
    # Calculate the variances
    var_conv = np.var(cdac.energy_conv)
    var_2step = np.var(cdac.energy_2step)
    var_cs = np.var(cdac.energy_cs)
    var_split = np.var(cdac.energy_split)
    
    # Print the variances
    np.set_printoptions(precision=5)
    print("Variance of Conventional:  {:.5f}".format(var_conv ))
    print("Variance of Two Step:       {:.5f}".format(var_2step))
    print("Variance of Charge Sharing: {:.5f}".format(var_cs   ))
    print("Variance of Split Cap:      {:.5f}".format(var_split))
    

def plot_temporal_2D(cdac):
    """
    Plots 8 2D graphs of the energy leakage over time for each switching 
    sequence using data from a CDAC object.
    
    Parameters:
    -----------
    cdac : CDAC object
        A CDAC object containing the data to plot.
    
    Returns:
    --------
    None
    """
            
    num_rows = cdac.conv_temporal.shape[1] - 1
    
    fig, axs = plt.subplots(num_rows, 1)
    
    
    for i in range(num_rows):
        # axs[i].step(np.arange(code_num), conv_temporal[:,i+1])# , '.')
        axs[i].step(np.arange(code_num), cdac.conv_temporal[:,i+1])
        axs[i].set_title(f"bit {bits-i}")
    
    plt.show()

def plot_temporal_3D(cdac):
    """
    Plots a 3D graph of the energy leakage over time and output code for all 
    curves from figure 2 using data from a CDAC object.
    
    Parameters:
    -----------
    cdac : CDAC object
        A CDAC object containing the data to plot.
    
    Returns:
    --------
    None
    """
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    num_rows = cdac.conv_temporal.shape[1] - 1
    for i in range(num_rows):
        zero_mean = cdac.conv_temporal[:,i+1] - np.mean(cdac.conv_temporal[:,i+1])
        ax.plot(np.arange(code_num), np.ones(code_num)*i, zero_mean, '-')
    
        # trying to make shadows for improved visibility
        x = np.arange(code_num)
        y = np.ones(code_num)*i + zero_mean / 20
        z = np.ones(code_num) * -33
        ax.plot(x, y, z, color='gray')
    ax.set_xlabel("Output Code")
    ax.set_ylabel("time (clock cycles)")
    ax.set_zlabel("Energy leakage (C$_0$V$^2$)")

def plot_temporal_3D_surface(cdac):
    """
    Plots a 3D surface of energy leakage over time and output code using data 
    from a CDAC object.
    
    Parameters:
    -----------
    cdac : CDAC object
        A CDAC object containing the data to plot.
    
    Returns:
    --------
    None
    """
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the data as a surface
    x, y = np.meshgrid(np.arange(cdac.split_temporal.shape[1]), np.arange(cdac.split_temporal.shape[0]))
    ax.plot_surface(x, y, cdac.split_temporal, cmap=plt.cm.Spectral,
                           linewidth=0, antialiased=False, rstride=1, cstride=1)
    ax.set_ylabel("Output Code")
    ax.set_xlabel("time (clock cycles)")
    ax.set_zlabel("Energy leakage (C$_0$V$^2$)")
    
    # plot as wireframe
    ##fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.plot_surface(x, y, split_temporal, drawstyle='steps')

def plot_1_heatmap(cdac):
    """
    Plots a single heatmap of the temporal energies of the conventional CDAC.

    Parameters:
    cdac (CDAC object): An object containing the CDAC data.

    Returns:
    None
    """
    
    # plot as heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(cdac.conv_temporal[:,1:], aspect='auto', interpolation='none', cmap='magma',
                   norm=LogNorm())
    
    #im = ax.imshow(split_temporal[:,1:], aspect='auto', interpolation='none', cmap='magma',
    #               vmin=None, vmax=None)
    
    # Add x and y axis
    # ax.set_xticks(np.arange(conv_temporal.shape[1]))
    # ax.set_yticks(np.arange(conv_temporal.shape[0]))
    
    # Add x and y axis labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Digital Code')
    
    # Add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Energy (C$_0$V$^2$)', rotation=90, labelpad=15)
    
    plt.show()

def plot_all_heatmaps(cdac):
    """
     Creates a figure with four heatmaps that show the energy values for each 
     cycle and digital code of a conventional CDAC, a 2-step CDAC, a charge 
     sharing CDAC, and a split-MSB CDAC. It then adds a colorbar to all 
     subplots and saves the figure as an EPS file.
    
     Parameters:
     -----------
     cdac : instance of the CDAC class
         The CDAC object containing the energy values to be plotted.
    
     Returns:
     --------
     None
     """
    # Save font size so we can keep it after finishing this plot
    font_size = plt.rcParams['font.size']
    
    # for the four vertical plots to mitigate scaling issues
    plt.rc('font', family='serif', size=font_size/2)
    plt.rc('xtick', labelsize=font_size/2)
    plt.rc('ytick', labelsize=font_size/2)
    
    # plot all 4
    fig, axs = plt.subplots(nrows=4, ncols=1)# , figsize=(3.5, 6))
    fig.subplots_adjust(hspace=0.4)
    
    x = np.linspace(1, 8, cdac.conv_temporal.shape[1])
    
    norm = LogNorm()

    # Using np.repeat to "upscale" the image so pdf viewers don't try to
    # interpolate the image which results in a blur effect
    im = axs[0].imshow(np.repeat(np.flipud(cdac.conv_temporal[:,1:]), 20, axis=1), 
                       aspect='.015', interpolation='none', cmap='magma', 
                       extent=[x[0], x[-1], 0, cdac.conv_temporal.shape[0]], norm=norm)
    #axs[0].set_title("Conventional CDAC")
    
    axs[1].imshow(np.repeat(np.flipud(cdac.step_temporal[:,1:]), 20, axis=1), 
                  aspect='.015', interpolation='none', cmap='magma', 
                  extent=[x[0], x[-1], 0, cdac.conv_temporal.shape[0]], norm=norm)
    #axs[1].set_title("2 Step CDAC")
    
    axs[2].imshow(np.repeat(np.flipud(cdac.cs_temporal[:,1:]), 20, axis=1), 
                  aspect='.015', interpolation='none', cmap='magma', 
                  extent=[x[0], x[-1], 0, cdac.conv_temporal.shape[0]], norm=norm)
    #axs[2].set_title("Charge Sharing CDAC")
    
    axs[3].imshow(np.repeat(np.flipud(cdac.split_temporal[:,1:]), 20, axis=1), 
                  aspect='.015', interpolation='none', cmap='magma', 
                  extent=[x[0], x[-1], 0, cdac.conv_temporal.shape[0]], norm=norm)
    #axs[3].set_title("Split-MSB CDAC")
    
    for ax in axs.flat:
        ax.set_xticks(np.arange(7)+1)
        ax.set_yticks([0, 128, 255])
    #    ax.set_xlabel('Time')
    #    ax.set_ylabel('Digital Code')
    
    cbar = fig.colorbar(im, ax=axs.ravel().tolist()) # add a colorbar to figure
    cbar.set_label('Energy (C$_0$V$^2$)', rotation=90, labelpad=5)
    cbar.ax.set_position([0.755, 0.1, 0.05, 0.8])
    plt.show()
    
    # Set tight layout
    # fig.tight_layout()
    
    # Save as EPS file
    plt.savefig('energies_temporal.eps', format='eps', bbox_inches='tight')
    
    # Restore previous font size
    plt.rc('font', size=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)


def plot_temporal_scatter(cdac):
    """
    Generates a scatterplot of the CDAC output values over time.
    
    Parameters:
        cdac (CDAC): The CDAC object that contains the CDAC output values.
    
    Returns:
        None.
    """
    plt.figure(7)
    plt.plot(cdac.conv_temporal.T, '.')
    plt.title('scatterplot of values over time')
    plt.ylabel('Energy (C$_0$V$^2$)')
    plt.xlabel('cycles')
    '''
    time_var = np.var(conv_temporal, axis=0)
    # print("Variance of Conventional: " + str(time_var))
    '''

def replicate_closed_form(cdac):
    """
    Replicates a signal in closed form and compares it with the conventional 
    CDAC output.
    
    Parameters:
        cdac (CDAC): The CDAC object that contains the conventional CDAC output.
    
    Returns:
        None.
    """
    from scipy import signal 
    t = np.arange(256)
    
    # this is a perfect replica with the parameter being 8 bits
    s = signal.square(np.pi * t / 64) * 8 * (1 + 2 * np.floor(t/128))
    
    plt.figure(9)
    plt.plot(s)
    plt.plot(cdac.conv_temporal[:,2] - np.mean(cdac.conv_temporal[:,2]))
    plt.title('replicating signal in closed form')
    
    # covariance matrix is diagonal:
    '''
    cov_split = np.cov(cdac.split_temporal, rowvar=False)
    cov_conv = np.cov(cdac.conv_temporal, rowvar=False)

    plt.figure()
    plt.imshow(cov_conv==0) 
    # covariance is diagonal and implies no linear correlation between switching steps

    # covariance matrix is diagonal => traces are linearly independent along time
    # however there is obviously a nonlinear dependence for which information could be extracted by a NN
    '''

def generate_variance_table(cdac):
    """
    Generates a table of variances for different CDAC architectures and prints 
    it to the console.

    Args:
        cdac (CDAC): An instance of the CDAC class that contains the temporal 
        switching energies for various CDAC architectures.

    Returns:
        None
    """
    
    print("Generating table of variances:\n")
    
    Conventional = np.var(cdac.conv_temporal[:,1:], axis=0)
    #print("Variance of Conventional: " + str(Conventional))
    Split_MSB = np.var(cdac.split_temporal[:,1:], axis=0)
    #print("Variance of Conventional: " + str(Split_MSB))
    Charge_Sharing = np.var(cdac.cs_temporal[:,1:], axis=0)
    #print("Variance of Conventional: " + str(Charge_Sharing))
    Two_Step = np.var(cdac.step_temporal[:,1:], axis=0)
    #print("Variance of Conventional: " + str(Two_Step))
    
    data = [
        ['Conventional'] + [float(np.sum(Conventional))] + Conventional.tolist() ,
        ['Two Step'] + [float(np.sum(Two_Step))] + Two_Step.tolist(),
        ['Charge Sharing'] + [float(np.sum(Charge_Sharing))] + Charge_Sharing.tolist(),
        ['Split MSB'] + [float(np.sum(Split_MSB))] + Split_MSB.tolist(),
    ]
    
    # Define the headers for the table
    headers = [''] + ['Total'] + list(np.arange(7)+1)
    
    # Convert the data to a list of lists
    table = list(data)
    
    # Add the headers to the table
    table.insert(0, headers)
    
    # Print the table as a LaTeX tabular environment
    print(tabulate(table, headers='firstrow', tablefmt='latex', floatfmt=".4g"))


if __name__ == '__main__':
    """
    Sets up plotting parameters and generates various plots related to CDAC 
    switching energies.
    """
    # set up to generate plots that look lice in LaTeX
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    plt.close('all')
    
    bits = 8            # bits of CDAC
    Vref = 1            # reference voltage
    C0 = 1              # unit capacitance
    
    code_num = 2**bits  # how many codes with given number of bits
    
    # get the cdac switching energies
    cdac = CDAC(bits, Vref, C0)
    cdac.calculate_energies()
    
    
    '''comment out unwanted plots'''
    plot_total_energy(cdac)
    # print_variances(cdac)
    plot_temporal_2D(cdac)
    plot_temporal_3D(cdac)
    plot_temporal_3D_surface(cdac)
    plot_1_heatmap(cdac)
    plot_all_heatmaps(cdac)
    plot_temporal_scatter(cdac)
    # replicate_closed_form(cdac)
    generate_variance_table(cdac)
    

