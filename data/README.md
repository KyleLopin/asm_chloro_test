Contains the data used for the analysis section of the project.

chlorophyll_data contains the raw data in the collected_data folder with 3 measurements per leaf, the data with the outliers removed that was used for analysis are in the pruned_data folder, the summary_data folder has the data with all the data points for each leaf averaged together.
Each leaf had its total chlorophyll, chlorophyll a and chlorophyll b measured 3 times.

hardare_test_data contains excel files of the AS7265x sensors LED_DRV pin voltage to show the UV LED doesn't not maintain proper voltage control.

spectrum_data contains the data for the sensor reads for the 3 sensors (as7262, as7263 and as7265x) and 5 leaves (banan, jasmine, mango, rice, sugarcane) for 15 data files.  These are summary files as each days files had 6-10 leaves and these were joined into a single file.
spectrum_data/raw_data is the raw data counts and spectrum_data/reflectance_data contains reflectance data with the reference spectrum divided by the raw counts for each integration time, led and led current condition (except for some 50 mA and 100 mA conditions were the references cause saturation, in those causes the other references are averaged to make a psuedo reference)
