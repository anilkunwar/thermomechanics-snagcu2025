import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_specific_enthalpy(df, element_composition):
    # Calculate molar weight of alloy or element
    molar_weights = {'Sn': 118.71, 'Ag': 107.8682, 'Cu': 63.546}  # Can be updated with molar weights of any elements present in the alloy (Sn, Ag, Cu )
    molar_weight = sum(element_composition[element] * molar_weights[element] for element in element_composition)

    # Calculate specific enthalpy (h) in J/kg
    df['h'] = df['H'] / molar_weight * 1000  # Convert from J/mol to J/kg
    return df[['T', 'h']]

def save_results(df, filename, format='csv'):
    if format == 'csv':
        df.to_csv(filename, index=False)
    elif format == 'dat':
        np.savetxt(filename, df.values, fmt='%.5f', header='T (K), h (J/kg)', comments='')
    else:
        st.error("Unsupported output format. Please choose 'csv' or 'dat'.")

def main():
    st.title("Specific Enthalpy Calculation")

    # Upload CSV file
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.subheader("Uploaded Data")
        st.write(df)

        # Get alloy composition from user
        st.header("Alloy Composition")
        elements = ['Sn', 'Ag', 'Cu']  # Sn, Ag, Cu
        element_composition = {}
        for element in elements:
            mole_fraction = st.number_input(f"Mole Fraction of {element}", min_value=0.0, max_value=1.0, value=0.33333)
            element_composition[element] = mole_fraction

        # Calculate specific enthalpy
        specific_enthalpy_df = calculate_specific_enthalpy(df, element_composition)

        # Plot H vs. T and h vs. T curves
        st.header("Plots")
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))

        # Plot H vs. T curve
        axes[0].plot(df['T'], df['H'], label='H vs. T')
        axes[0].set_xlabel('Temperature (K)')
        axes[0].set_ylabel('Enthalpy (J/mol)')
        axes[0].set_title('Enthalpy vs. Temperature')
        axes[0].legend()

        # Plot h vs. T curve
        axes[1].plot(specific_enthalpy_df['T'], specific_enthalpy_df['h'], label='h vs. T')
        axes[1].set_xlabel('Temperature (K)')
        axes[1].set_ylabel('Specific Enthalpy (J/kg)')
        axes[1].set_title('Specific Enthalpy vs. Temperature')
        axes[1].legend()

        # Display plots
        st.pyplot(fig)

        # Save results
        st.header("Save Results")
        output_format = st.radio("Select output format", options=['csv', 'dat'])
        output_filename = st.text_input("Enter output filename", value="output")

        save_button = st.button("Save Results")
        if save_button:
            save_results(specific_enthalpy_df, f"{output_filename}.{output_format}", format=output_format)
            st.success("Results saved successfully.")

if __name__ == "__main__":
    main()
