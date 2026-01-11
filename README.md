# thermomechanics-snagcu2025

1. Computation of Enthalpy
   
   (a) temperature_molar-enthalpy.csv
   The csf file consists of the following two columns:
   (i) temperature T (K)
   (ii) Molar enthalpy H (J/mol) of Sn-3.0Ag0.5Cu Alloy. The wt. % of Ag and Cu in the alloy are 3.0 % and 0.5 %.
   So, w_Sn = 96.5 %, w_Ag = 3.0 % and w_Cu = 0.5 %.
   In terms of mole fraction, x_Sn = 0.9580, x_Ag = 0.0327 and x_Cu = 0.0093.
    

   This data is computed from a TDB file using pycalphad software.  A webapp (https://enthalpydatafromtdbfile.streamlit.app/) has been constructed to make this computational task easier.
   The  thermophysical quantities such as  melting point and  latent heat of fusion for the alloy can be estimated for the Sn-xAg-yCu alloy by fitting the T-H data in another webapp (https://enthalpytemperature.streamlit.app/) 

   (b) computation of specific enthalpy of Sn-3.0Ag0.5Cu alloy from the molar enthalpy values
   
   As h (J/kg) is utilized in the finite element analysis, it is necessary to compute the h-T data.
   (i) code : specific_enthalpy_from_molar_enthalpy.py
   (ii) generated data: T_h-SnAgCu.csv (csv format) and T_h-SnAgCu.dat (dat file format)


2. Attention Mechanism

   Reading the simulation results
   
   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutionreader1-streamlit-red)](https://simulation-files-reader1.streamlit.app/)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutionreader2-streamlit-red)](https://simulation-files-reader2.streamlit.app/)


   Interpolating (Extrapolating) the solutions

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator1-streamlit-red)](https://thermomechanical-solutions-interpolator1.streamlit.app/)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator2-streamlit-red)](https://thermomechanical-solutions-interpolator2.streamlit.app/)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator3-streamlit-red)](https://thermomechanical-solutions-interpolator3.streamlit.app/) (attention weights)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator4-streamlit-red)](https://thermomechanical-solutions-interpolator4.streamlit.app/)  (more robust)

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator5-streamlit-red)](https://thermomechanical-solutions-interpolator5.streamlit.app/) (visualization of extrapolated/interpolated results)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator6-streamlit-red)](https://thermomechanical-solutions-interpolator6.streamlit.app/)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator7-streamlit-red)](https://thermomechanical-solutions-interpolator7.streamlit.app/) (data loading in interpolation module not yet resolved for entire mesh and geometry, unrealistic interpolation )
   
   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator8-streamlit-red)](https://thermomechanical-solutions-interpolator8.streamlit.app/) (developed from the model r4)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator9-streamlit-red)](https://thermomechanical-solutions-interpolator9.streamlit.app/) (developed from the model r4)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator10-streamlit-red)](https://thermomechanical-solutions-interpolator10.streamlit.app/) (developed from the model r4)

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator11-streamlit-red)](https://thermomechanical-solutions-interpolator11.streamlit.app/) (robust and working interpolation module)

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator12-streamlit-red)](https://thermomechanical-solutions-interpolator12.streamlit.app/) 
