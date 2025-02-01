# thermomechanics-snagcu2025

1. Themodynamic Dataset
   
   (a) temperature_molar-enthalpy.csv
   The csf file consists of the following two columns:
   (i) temperature T (K)
   (ii) Molar enthalpy H (J/mol) of Sn-3.0Ag0.5Cu Alloy. The wt. % of Ag and Cu in the alloy are 3.0 % and 0.5 %.
   So, w_Sn = 96.5 %, w_Ag = 3.0 % and w_Cu = 0.5 %.
   In terms of mole fraction, x_Sn = 0.9580, x_Ag = 0.0327 and x_Cu = 0.0093.
    

   This data is computed from a TDB file using pycalphad software.  A webapp (https://enthalpydatafromtdbfile.streamlit.app/) has been constructed to make this computational task easier.

   (b) computing the specific enthalpy (h) data
   As h (J/kg) is utilized in the finite element analysis, it is necessary to compute the h-T data. 

