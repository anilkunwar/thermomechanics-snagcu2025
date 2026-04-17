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


2. Attention Mechanism for Contruction of Numerically Attentive Interpolation and Extrapolation Framework from Finite Element Fields Datasets

   Reading the simulation results
   
   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutionreader1-streamlit-red)](https://simulation-files-reader1.streamlit.app/)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutionreader2-streamlit-red)](https://simulation-files-reader2.streamlit.app/)


   Interpolating (Extrapolating) the solutions (works better with python 3.12 or earlier versions)

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

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator11-streamlit-red)](https://thermomechanical-solutions-interpolator11.streamlit.app/) (robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, Milestone 1)

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator12-streamlit-red)](https://thermomechanical-solutions-interpolator12.streamlit.app/) (robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, Milestone 1)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator13-streamlit-red)](https://thermomechanical-solutions-interpolator13.streamlit.app/) (robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, Milestone 1, r11 with a longer estimation duration)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator14-streamlit-red)](https://thermomechanical-solutions-interpolator14.streamlit.app/) (yet to be completed)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator15-streamlit-red)](https://thermomechanical-solutions-interpolator15.streamlit.app/) (gated attention with weights for flux and pulse duration, robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, Milestone 1, r11 with a longer estimation duration)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator16-streamlit-red)](https://thermomechanical-solutions-interpolator16.streamlit.app/) (gated attention with weights for flux, pulse duration and time, robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, Milestone 1, r11 with a longer estimation duration)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator17-streamlit-red)](https://thermomechanical-solutions-interpolator17.streamlit.app/) (gated attention with weights for flux, pulse duration and time, robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, Milestone 1, r11 with a longer estimation duration, opacity of dataviewer changeable for better visualization in only the dataviewer section, render detailed results function corrected)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator18-streamlit-red)](https://thermomechanical-solutions-interpolator18.streamlit.app/) (gated attention with weights for flux, pulse duration and time, robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, Milestone 1, r11 with a longer estimation duration, opacity of dataviewer changeable for better visualization in all places, render detailed results function corrected)


   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator19-streamlit-red)](https://thermomechanical-solutions-interpolator19.streamlit.app/) (gated attention with weights for flux, pulse duration and time, robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, opacity of dataviewer changeable for better visualization in all places, render detailed results function corrected, customized for experimental target E and tau values)

   FEM solutions visualization of fields

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutionviewer20-streamlit-red)](https://thermomechanical-solutions-data-viewer20.streamlit.app/) (customizable color bars, yet the color bars for each fields are not represented in customizable bar)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutionviewer21-streamlit-red)](https://thermomechanical-solutions-data-viewer21.streamlit.app/) (customizable color bars and controllable opacity, color bars edge values setting function and color maps perform correctly )

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutioninterpolator22-streamlit-red)](https://thermomechanical-solutions-interpolator22.streamlit.app/) (r19 with customizable color bars for 3D vision i.e. color bars edge values setting function and color maps perform correctly , gated attention with weights for flux, pulse duration and time, robust and working interpolation module, no temporal bias and so interpolation only within 20 ns duration, opacity of dataviewer changeable for better visualization in all places, render detailed results function corrected, customized for experimental target E and tau values)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/solutionviewer23-streamlit-red)](https://thermomechanical-solutions-data-viewer23.streamlit.app/) (customizable color bars and controllable opacity, color bars edge values setting function and color maps perform correctly, hierarchical radar/sunburst chart as solution viewer )
   
  
   Robust Visualizations 

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/sankeydemo-v1-brightgreen.svg)](https://thermomechanical-demo-data-interpolation-sankey-visualization1.streamlit.app/) (Sankey visualization for demo data)

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/sankeydemo-v2-brightgreen.svg)](https://thermomechanical-demo-data-interpolation-sankey-visualization2.streamlit.app/) (Sankey visualization for demo data)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/sankey-v3-brightgreen.svg)](https://thermomechanical-solutions-interpolation-sankey-visualization3.streamlit.app/) (Sankey visualization for FEM simulations data, Random 10 weightage based most relevant source simulations are visualized instead of the total 23*8 = 184 i.e. E tau steady state simulations * number of timesteps, Data viewer part needs improvement)

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/sankey-v4-brightgreen.svg)](https://thermomechanical-solutions-interpolation-sankey-visualization4.streamlit.app/) (Sankey visualization for FEM simulations data, incomplete, Top N (5<=N<=184 steady state simulations) weightage based most relevant source simulations are visualized from  the total 23*8 = 184 i.e. E tau steady state simulations * number of timesteps, Data viewer part needs improvement)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/polardiagram-v5-brightgreen.svg)](https://thermomechanical-solutions-interpolation-visualization-r5.streamlit.app/) (Radar chart visualization for FEM simulations data)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/polardiagram-v6-brightgreen.svg)](https://thermomechanical-solutions-interpolation-visualization-r6.streamlit.app/) (Radar chart visualization for FEM simulations data, enhanced and robust visualization)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/polardiagram-v7-brightgreen.svg)](https://thermomechanical-solutions-interpolation-visualization-r7.streamlit.app/) (Radar chart visualization for FEM simulations data,st-dgpa visualization is enhanced, radar chart is empty )

    [![continuummodellaserprocessing3d](https://img.shields.io/badge/polardiagram-v8-brightgreen.svg)](https://thermomechanical-solutions-interpolation-visualization-r8.streamlit.app/) (Radar chart visualization for FEM simulations data,st-dgpa visualization is enhanced, radar chart is complete, normalized chart needs to be transformed to physics units )

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/sankey-v9-brightgreen.svg)](https://thermomechanical-solutions-interpolation-sankey-visualization9.streamlit.app/) (Sankey visualization for FEM simulations data, incomplete, Random N (5<=N<=184 steady state simulations) weightage based most relevant source simulations are visualized from  the total 23*8 = 184 i.e. E tau steady state simulations * number of timesteps, Data viewer part needs improvement, contd from r4, consistency check of Sankey and the csv data)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/sankey-v10-brightgreen.svg)](https://thermomechanical-solutions-interpolation-sankey-visualization10.streamlit.app/) (Sankey visualization for FEM simulations data, incomplete, Top N, Bottom N or Random N (5<=N<=184 steady state simulations) weightage based most relevant source simulations are visualized from  the total 23*8 = 184 i.e. E tau steady state simulations * number of timesteps, Data viewer part needs improvement, contd. from r4, consistency check of Sankey and the csv data)

   [![continuummodellaserprocessing3d](https://img.shields.io/badge/polardiagram-v11-brightgreen.svg)](https://thermomechanical-solutions-interpolation-visualization-r11.streamlit.app/) (Radar chart visualization for FEM simulations data,st-dgpa visualization is enhanced, radar chart is complete, normalized chart ... transformed to physics units )
   
   
   
   
