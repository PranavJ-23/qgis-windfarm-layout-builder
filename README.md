# Layout Builder for Onshore Windfarms

This QGIS plugin helps automate the layout design for onshore wind farms in Europe by fetching and analysing wind data to generate an optimised turbine layout.

## Features

*   **Wind Data Integration:** Fetches wind statistics and time series data from the New European Wind Atlas (NEWA).
*   **Wind Analysis:** Calculates the dominant wind direction for the selected location.
*   **Wind Rose Visualisation:** Generates and displays a  wind rose to visualise wind speed and direction distributions.
*   **Optimised Layout Generation:** Uses a multi-strategy hexagonal packing algorithm to place the maximum number of turbines within a specified polygonal area.
*   **Wake Effect Consideration:** The packing algorithm considers the wake effect of turbines by using elliptical exclusion zones.
*   **Shapefile Export:** Exports the generated turbine locations as a point shapefile and the wake ellipses as a polygon shapefile.

## Installation

This plugin is self-contained and includes all necessary dependencies.

1.  Download the `Layout Builder for Onshore Windfarms.zip` file from the [GitHub Releases](https://github.com/PranavJ-23/qgis-windfarm-layout-builder/releases) page.
2.  Open QGIS.
3.  Navigate to **Plugins > Manage and Install Plugins...**.
4.  Go to the **Install from ZIP** tab.
5.  Click the `...` button and select the `.zip` file you downloaded.
6.  Click **Install Plugin**.

The plugin will be installed and will appear in the QGIS Vector menu and as an icon in the toolbar.

## Requirements

*   QGIS 3.0 or higher.

All Python dependencies are included in the plugin package.

## Usage

1.  Click on the plugin icon in the toolbar or find it in the **Vector > Layout Builder for Onshore Windfarms** menu.
2.  **Fetch Wind Data:**
    *   Enter the latitude and longitude for your area of interest, or select a point layer from your project.
    *   Click **Initialize Wind Data**.
3.  **Generate Layout:**
    *   Select a polygon layer that defines the construction area.
    *   Set the turbine rotor diameter, grid spacing, and number of iterations for the packing algorithm.
    *   Click **Generate Layout**. The process will run in the background.
4.  **Save Results:** Once the layout is generated, you will be prompted to save the turbine locations as a shapefile. You can also choose to export the wake ellipses.
5.  **Plot Wind Rose:** After fetching wind data, click **Plot Wind Rose** to view a detailed wind rose image for your location. You can save this image as a JPG.

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue on the [GitHub repository](https://github.com/PranavJ-23/qgis-windfarm-layout-builder).

## License

This plugin is licensed under the **GNU General Public License v2.0 (or any later version)**.