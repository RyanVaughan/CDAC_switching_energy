# CDAC_switching_energy
modeling the switching energy of a CDAC as used in a SAR ADC. Multiple architectures are considered. Result can be used to analyze power side channel leakage.

# Getting Started
cdac.py contains the CDAC class which is used to simulate all of the switching energy.
plot.py contains example plots to visualize the data
  plot.py can be run which will execute the code under if __name__ == '__main__':
  by default, it generates all of the plots setup for LaTeX.
  The two plots "plot_total_energy" and "plot_all_heatmaps" also save the result as a .eps file
  generate_variance_table generates the LaTeX table containing the variances of the four architectures
