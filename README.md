## Vortex leapfrogging and superfluid dissipation mechanisms in a fluid of light

### Codes:
- contrast.py contains all image density and phase extraction from raw data
- velocity.py to extract information from fields array
- data_processing.py to run the treatement

### Description of the Dataset
Download the data on Zenodo: [link](????)

This dataset contains numerical and experimental data used to generate the figures in our article. The provided files include raw interferograms, processed field data, and computed quantities such as energy spectra and velocity fields.

#### **Figures and Data Sources**

-   **Fig1**: Data from `04302233_sheep_feedbackloop`
-   **Fig2**: Data from `03201801_sheep_jrs`
-   **Fig3**: Data from `04151542_sheep_big`

#### **Field Data**

The `field.npy` files contain 4D NumPy arrays with shape `(i, j, Ny, Nx)`, where:

-   `i` represents the number of time steps,
    
-   `j` corresponds to the number of images averaged at each time step,
    
-   `Ny, Nx` are the spatial dimensions.
    
-   `field_ref.npy` corresponds to a reference Gaussian field without vortices.
    
-   `field_vortex.npy` contains fluid data with a single vortex, used to measure vortex size at each time step.
    

#### **Computed Quantities**

The dataset also includes various derived quantities, such as energy distributions and velocity fields, vortex trajectories...

This dataset is openly available for further analysis and verification.
