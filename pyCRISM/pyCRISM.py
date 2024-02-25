#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## pyCRISM.py
## Created by Binlong Ye
## Last modified by Binlong Ye : 09/01/2024

##----------------------------------------------------------------------------------------
"""
A python implementation of CAT to process CRISM data
"""
##----------------------------------------------------------------------------------------
import spectral.io.envi as envi
import numpy as np
import pdr
import os
from osgeo import gdal
import subprocess
from scipy.ndimage import uniform_filter1d

# Name of the current file
_py_file = 'pyCRISM.py'
_Version = 1.0
package_path = os.path.abspath(os.path.dirname(__file__))

class CRISMdata:
    def __init__(self, input_file):
        np.seterr(divide='ignore', invalid='ignore')
        self.setup_paths(input_file)
        if not self.check_files_exist():
            return
        self.metadata = self.load_metadata()
        if self.metadata is None:
            print("Failed to load metadata.")
            return
        self.process_metadata()
        self.geomdata = np.array([])
        self.ifdat = np.array([])
        self.wvlc = np.array([])
        self.wvlc_at = np.array([])
        self.atm = np.array([])
        self.ifdat_corr = np.array([])
        self.meanwvl = np.array([])
        self.resample = False
        self.Photometric_correction = False
        self.Atmospheric_correction = False
        self.summary_product = None
    def display_metadata(self):
        print("CRISM Data Metadata:")
        print(f"pyCRISM_Version: {_Version}")
        print(f"File Name: {self.file_name}")
        print(f"Size (Y, X, L): ({self.size_y}, {self.size_x}, {self.size_l})")
        print(f"Solar Longitude: {self.ls}")
        print(f"Wavelength File: {self.data_wvl_file}")
        print(f"Distance to Mars: {self.dmars} AU")
        print(f"Channel: {self.channel}")
        print(f"Pixel Averaging Width: {self.binning}")
        print(f"Observation Name: {self.obs_name}")
        print(f"Start Time: {self.start_time}")
        print(f"Detector Temperature: {self.detector_temp}")
        print(f"Photometric Correction: {self.Photometric_correction}")
        print(f"Atmospheric Correction: {self.Atmospheric_correction}")
        print(f"Resampled: {self.resample}")
    def __str__(self):
        metadata_str = "CRISM Data Metadata:\n"
        metadata_str = f"pyCRISM_Version: {_Version}\n"
        metadata_str += f"File Name: {self.file_name}\n"
        metadata_str += f"Size (Y, X, L): ({self.size_y}, {self.size_x}, {self.size_l})\n"
        metadata_str += f"Solar Longitude: {self.ls}\n"
        metadata_str += f"Wavelength File: {self.data_wvl_file}\n"
        metadata_str += f"Distance to Mars: {self.dmars} AU\n"
        metadata_str += f"Channel: {self.channel}\n"
        metadata_str += f"Pixel Averaging Width: {self.binning}\n"
        metadata_str += f"Observation Name: {self.obs_name}\n"
        metadata_str += f"Start Time: {self.start_time}\n"
        metadata_str += f"Detector Temperature: {self.detector_temp}\n"
        metadata_str += f"Photometric Correction: {self.Photometric_correction}\n"
        metadata_str += f"Atmospheric Correction: {self.Atmospheric_correction}\n"
        metadata_str += f"Resampled: {self.resample}\n"
        return metadata_str    
    def __repr__(self):
        return self.__str__()

    def setup_paths(self, input_file):
        """
        Setup the necessary file paths for CRISM data and metadata.
        """
        self.input_folder = input_file[:-30]
        crism_file_name = input_file[-30:-4]
        self.ifdat_path = input_file[:-4] + '.img'
        self.ifdat_lbl_path = input_file[:-4] + '.lbl'
        self.geomdata_path = os.path.join(self.input_folder, crism_file_name[:15] + 'de' + crism_file_name[17:22] + 'ddr1.img')
        
    def check_files_exist(self):
        """
        Check if all necessary files exist.
        """
        for path in [self.ifdat_path, self.ifdat_lbl_path, self.geomdata_path]:
            if not os.path.exists(path):
                print(f"File not found: {path}")
                return False
        return True
    def read_crism_lbl(self,lblfile,keywords):
        """
        Reads a CRISM LBL file and extracts specified parameters.
        """
        values = {}
        with open(lblfile, 'r') as file:
            for line in file:
                if line.strip() == 'END':
                    break
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')  # Remove potential quotes
                    if key in keywords:
                        values[key] = value
        return values
    def convert_tif_to_envi(self, input_tif, output_envi):
        command = ['gdal_translate', '-of', 'ENVI', input_tif, output_envi]
        subprocess.run(command, check=True)
    
    def load_metadata(self):
        """
        Load metadata from the label file.
        """
        keywords = ['RECORD_BYTES', 'FILE_RECORDS', 'LINES', 'LINE_SAMPLES', 'BANDS',
                    'SOLAR_LONGITUDE', 'MRO:WAVELENGTH_FILE_NAME', 'SOLAR_DISTANCE',
                    'MRO:SENSOR_ID', 'MRO:WAVELENGTH_FILTER', 'PIXEL_AVERAGING_WIDTH',
                    'SPACECRAFT_CLOCK_START_COUNT', 'PRODUCT_ID', 'MRO:DETECTOR_TEMPERATURE',
                    'MRO:OPTICAL_BENCH_TEMPERATURE', 'MRO:SPECTROMETER_HOUSING_TEMP', 'START_TIME']
        try:
            # Assuming read_crism_lbl is implemented to read key-value pairs from the LBL file
            return self.read_crism_lbl(self.ifdat_lbl_path,keywords)
        except FileNotFoundError:
            print(f"File not found: {self.ifdat_lbl_path}")
            return None

    def process_metadata(self):
        """
        Process and store metadata from the label file.
        """
        values = self.metadata
        self.size_y = int(values['FILE_RECORDS'])
        self.size_x = int(values['LINE_SAMPLES'])
        self.size_l = int(values['BANDS'])
        self.ls = float(values['SOLAR_LONGITUDE'])
        self.data_wvl_file = values['MRO:WAVELENGTH_FILE_NAME']
        self.dmars = float(values['SOLAR_DISTANCE'][:16]) / 149598000.
        self.channel = values['MRO:SENSOR_ID']
        self.wvlfilter = values['MRO:WAVELENGTH_FILTER']
        self.binning = int(values['PIXEL_AVERAGING_WIDTH'])
        self.file_name = values['PRODUCT_ID']
        self.obs_name = self.file_name.split('_')[0]
        self.sc_clock = values['SPACECRAFT_CLOCK_START_COUNT']
        self.bin_modes = {1: '0', 2: '1', 5: '2', 10: '3'}
        self.binmode = self.bin_modes.get(self.binning, 'unknown')
        self.obs_type = self.obs_name[:3]
        self.detector_temp = float(values['MRO:DETECTOR_TEMPERATURE'])
        self.bench_temp = float(values['MRO:OPTICAL_BENCH_TEMPERATURE'])
        self.spechousing_temp = float(values['MRO:SPECTROMETER_HOUSING_TEMP'])
        self.start_time = values['START_TIME']
    def load(self):
        """
        Load and process CRISM image data.
        """
        try:
            # Assuming pdr.read is a function to read the image data
            self.ifdat = self.load_and_process_image(self.ifdat_path)
            self.ifdat = self.ifdat[:,:,::-1]
            self.geomdata = self.load_and_process_image(self.geomdata_path)
            self.load_wavelength_data()
            self.load_atmosphere_wavelength_data()   
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None
    def load_wavelength_data(self):
        """
        Load and process wavelength data.
        """
        data_wvl_file = self.metadata['MRO:WAVELENGTH_FILE_NAME']
        wvlc_path = os.path.join(package_path,'aux_files','WA', data_wvl_file)
        self.wvlc = pdr.read(wvlc_path)
        self.wvlc = self.wvlc['IMAGE']
        self.wvlc = np.transpose(self.wvlc, axes=(1,2,0))
        self.wvlc *= 0.001
        self.wvlc[self.wvlc == 65.5350] = 0.0
        self.wvlc = self.wvlc[:, :, ::-1]
        self.wvlc = self.wvlc[0,:,:]
        self.meanwvl = self.mean_wavelength()
        self.meanwvl[-1]=4.0
    def load_atmosphere_wavelength_data(self):
        """
        Load and process atmospheric wavelength data.
        """
        sample_name = 0  # Default value, adjust as necessary
        band_name = 0
        if(self.size_x == 640):
            sample_name=0
        if(self.size_x== 320):
            sample_name=1
        if (self.size_l == 438):
            band_name=0
        if (self.size_l== 55):
            band_name=1
        if (self.size_l == 70):
            band_name=2
        if (self.size_l == 62):
            band_name=3
        crism_obs_info = os.path.join(package_path,'aux_files/crism_obs_info_resize.txt')
        with open(crism_obs_info, "r") as file:
            for line in file:
                if line[1:9] == self.obs_name[3:11]:
                    vs_id = line
        vs_id = str(vs_id[64:69])

        vs_lbl= os.path.join(package_path,'aux_files/VS_ADR/ADR10000000000_' + vs_id + "_VS"+str(sample_name)+str(band_name)+"L_8.LBL")
        vs_img= os.path.join(package_path,'aux_files/VS_ADR/ADR10000000000_'+vs_id + "_VS"+str(sample_name)+str(band_name)+"L_8.IMG")

        atmlbl = self.read_crism_lbl(vs_lbl, ['MRO:WAVELENGTH_FILE_NAME'])
        at_wvl_file = atmlbl['MRO:WAVELENGTH_FILE_NAME']
        wvlc_at_path = os.path.join(package_path,'aux_files', 'WA', at_wvl_file)
        self.wvlc_at = pdr.read(wvlc_at_path)
        self.wvlc_at = self.wvlc_at['IMAGE']
        self.wvlc_at = np.transpose(self.wvlc_at, axes=(1,2,0))
        self.wvlc_at *= 0.001
        self.wvlc_at[self.wvlc_at == 65.5350] = 0.0
        self.wvlc_at = self.wvlc_at[:, :, ::-1]
        self.wvlc_at = self.wvlc_at[0,:,:]

        self.atm = pdr.read(vs_img)
        self.atm = np.transpose(self.atm['IMAGE'], axes=(1,2,0))
        self.atm = self.atm[:,:,::-1]
        self.atm[self.atm==65535] = 0

    def load_and_process_image(self, path):
        """
        Load and process a single CRISM image file.
        """
        data = pdr.read(path)  # Placeholder for actual data reading function
        data = np.transpose(data['IMAGE'], axes=(1, 2, 0))
        data[data == 65535] = 0  # Example of a specific data correction
        #data = data[:, :, ::-1]  # Reverse the wavelength axis
        return data
    def crism_corr(self):
        """
        Apply corrections including photometric, atmospheric, VS artifact patching,
        and a second atmospheric correction. Store the final corrected data in self.ifdat_corr.
        """
        print('<> Photometric Correction Processing')
        # Apply photometric correction and store in a temporary variable
        ifdat_photometric_corrected = self.crism_photometric_correction()
        self.Photometric_correction = True
        print('<> Atmospheric Correction Processing')
        # Perform the first atmospheric correction on the photometrically corrected data
        ifdat_atm_corrected = self.crism_atmospheric_correction(ifdat_photometric_corrected)
        ifdat_patched = self.patch_vs_artifact(ifdat_atm_corrected, self.atm)
        self.ifdat_corr = self.crism_atmospheric_correction(ifdat_patched)
        self.Atmospheric_correction = True        
        print('<> Correction Finished')

    def crism_photometric_correction(self):
        """
        Apply photometric correction to self.ifdat and return the corrected data.
        """
        incidence = self.geomdata[:, :, 0]
        incidence = 1.0 / np.cos(np.radians(incidence))
        incidence[~np.isfinite(incidence)] = 0
        incidence = np.convolve(incidence.ravel(), np.ones(3) / 3, mode='same').reshape(self.size_y, self.size_x)
        incidence = np.reshape(incidence, (self.size_y, self.size_x, 1))
        ifdat_pho_corr = self.ifdat * incidence
        ifdat_pho_corr[~np.isfinite(ifdat_pho_corr)] = np.nan
        return ifdat_pho_corr

    def crism_atmospheric_correction(self,ifdat):
        """
        Apply atmospheric correction using the McGuire 2 wavelengths method.

        Parameters:
            ifdat : numpy.ndarray
                The original spectral data array.
            wvlc : numpy.ndarray
                Wavelengths array corresponding to the spectral data.
            wvlc_at : numpy.ndarray
                Wavelengths array for atmospheric data.
            atm : numpy.ndarray
                Atmospheric transmission data.
            beta : numpy.ndarray
                The beta parameter for correction.
        Returns:
            ifdat_corr: numpy.ndarray
                The atmospherically corrected spectral data.
        """
        temp_atm = self.atm[0,:,:]
        temp_atm = temp_atm.reshape((1,self.size_x, self.size_l))

        beta = np.zeros((self.size_y, self.size_x))
        for i in range(self.size_x):
            r2007index = np.argmin(np.abs(self.wvlc[i, :] - 2.007))
            r1980index = np.argmin(np.abs(self.wvlc[i, :] - 1.980))
            r2007index_at = np.argmin(np.abs(self.wvlc_at[i, :] - 2.007))
            r1980index_at = np.argmin(np.abs(self.wvlc_at[i, :] - 1.980))
            with np.errstate(divide='ignore', invalid='ignore'):
                beta[:, i] = np.log(ifdat[:, i, r2007index] / ifdat[:, i, r1980index]) / np.log(
                    temp_atm[:, i, r2007index_at] / temp_atm[:, i, r1980index_at])

        temp_atm = temp_atm ** beta[:, :, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            ifdat_corr = ifdat / temp_atm
            ifdat_corr[~np.isfinite(ifdat_corr)] = np.nan
        return ifdat_corr

    def patch_vs_artifact(self,data,atm):
        data = data[:,:,::-1]
        data = np.array(data)
        # merit function parameters
        merit_filter = 3
        max_iterations = 3
        ns, nb = data.shape[1], data.shape[2]
        zero_deriv_count = 0  # Not used in the provided script
        iterations = 0
        b1, b2 = 249, 322
        avc = np.mean(data[:, :, 247:252], axis=2)
        avc = np.expand_dims(avc, axis=2)
        avc = np.concatenate((avc, np.expand_dims(np.mean(data[:, :, 320:324], axis=2),axis=2)), axis=2)
    
        avg_cont = np.mean(avc, axis=2)
        avg_cont = np.expand_dims(avg_cont, axis=2)
        nb = b2 - b1

        merit = 1.0e23
        dscl = 1.0e23
        scl_fac = 1.0
        iterations = 1
        reversed_atm = self.atm[:,:,::-1]
        art = reversed_atm[1,:,249:322].reshape(1,self.size_x,73)

        while iterations <= max_iterations:
            patch1 = data[:, :, b1:b2] + np.tile(avg_cont, (1, 1, nb)) * scl_fac * art
            merit = self.evaluate_catvs_patch(patch1, art, merit_filter)
            delta = abs(scl_fac) * 1.0e-3
            delta = np.array(delta)
            delta[delta < 0.0003] = 0.0003

            scl_fac2 = scl_fac + delta
            delta = scl_fac2 - scl_fac
            patch2 = data[:, :, b1:b2] + np.tile(avg_cont, (1, 1, nb)) * scl_fac2 * art
            merit2 = self.evaluate_catvs_patch(patch2, art, merit_filter)
            dmds = (merit2 - merit) / delta

            dscl = -merit / dmds
            scl_fac += dscl
            iterations += 1
        data[:,:,b1:b2]=data[:,:,b1:b2]+np.tile(avg_cont, (1, 1, nb)) * scl_fac * art
        return data[:, :,::-1]
    def evaluate_catvs_patch(self,patch, art, smw):
        # Assumption: smw is the size of the smoothing window
        art_filtered = art - uniform_filter1d(art, smw,axis=2)
        patch_filtered = patch - uniform_filter1d(patch, smw,axis=2)
        nb = art.shape[2]

        corr = np.sum(patch_filtered[:, :, 3:nb - 4] * art_filtered[:, :, 3:nb - 4], axis=2)
        corr = corr.reshape(corr.shape[0],corr.shape[1],1)
        return corr
   
    def mean_wavelength(self):
        """
        Calculate the mean wavelength for each pixel in the image.
        """
        wvlc = self.wvlc
        wvlc = wvlc[:, ::-1]
        meanwvl = np.zeros(self.size_l)
        meanwvl = np.mean(wvlc[int(260/self.binning):int(359/self.binning)], axis=0)
        meanwvl[1]=4.0
        return meanwvl[::-1]
    def crism_resample(self):
        for i in range(int(30 / self.binning), int(634 / self.binning)):
            data_row = self.ifdat_corr[:, i, 2:]
            wl_row = self.wvlc[i, 2:]
            interpolator = interp1d(wl_row, data_row, kind='linear', fill_value="extrapolate")
            resampled_data = interpolator(self.meanwvl[2:])
            self.ifdat_corr[:, i, 2:] = resampled_data
            self.resample = True
    def project_crism(self):
        ifdat = np.array(self.ifdat_corr)
        geocube = self.geomdata  # Assuming this is where your lat/lon data is stored
        latlon = geocube[:, :, 3:5]
        dlatlon = np.array([latlon.shape[1], latlon.shape[0]])
        centerlon = latlon[dlatlon[1] // 2, dlatlon[0] // 2, 1]
        ctrl_pts_file = os.path.join(self.input_folder, "ctrl_pts.txt")

        with open(ctrl_pts_file, "w") as file:
            for i in range(5, dlatlon[1] + 1, dlatlon[1] // 10):
                for j in range(5, dlatlon[0] + 1, dlatlon[0] // 10):
                    if i > dlatlon[1]:
                        i = dlatlon[1]
                    if j > dlatlon[0]:
                        j = dlatlon[0]
                    file.write(f"-gcp {j} {i} {latlon[i-1, j-1, 1]} {latlon[i-1, j-1, 0]}\n")

        rows, cols, bands = ifdat.shape
        driver = gdal.GetDriverByName('GTiff')
        temp_image = os.path.join(self.input_folder, "project_crism_temp_image.tif")
        dataset = driver.Create(temp_image, cols, rows, bands, gdal.GDT_Float32)

        for i in range(bands):
            band = dataset.GetRasterBand(i+1)
            band.WriteArray(ifdat[:, :, i])
            dataset.FlushCache()

        translated_image = os.path.join(self.input_folder, "project_crism_image.tif1")
        gdal_translate_cmd = f'gdal_translate -of GTIFF -a_srs "+proj=longlat +a=3396190 +b=3376200 +units=m +no_defs" `cat {ctrl_pts_file}` "{temp_image}" "{translated_image}"'
        subprocess.run(gdal_translate_cmd, shell=True, check=True)

        projected_image = os.path.join(self.input_folder, "project_crism_image.tif2")
        gdalwarp_cmd = (
            f'gdalwarp -of GTIFF -q -rb -srcnodata 0 '
            f'-t_srs "+proj=sinu +lon_0={centerlon} +x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs" '
            f'-tps "{translated_image}" "{projected_image}"'
        )
        subprocess.run(gdalwarp_cmd, shell=True)

        final_output = os.path.join(self.input_folder, str.lower(self.file_name)+'_CAT_corr_p.tif')
        os.rename(projected_image, final_output)
        os.remove(ctrl_pts_file)
        os.remove(temp_image)
        os.remove(translated_image)
        self.convert_tif_to_envi(final_output,final_output[:-4])
        self.update_hdr_file(final_output[:-4]+'.hdr')
        os.remove(final_output)
    def crism_corr_save(self, output_folder = None):
        if output_folder is None:
            output_folder = self.input_folder
        ifdat_corr_metadata = {
            'description': 'PyCAT Processing',
            'default bands': [233 , 78 , 13],
            'sensor type' : 'Unknown',
            'header offset': '0',
            'file type':'ENVI Standard',
            'wavelength units': 'Micrometers',
            'data ignore value': '6.55350000e+004',
            }
        ifdat_corr_metadata['bands'] = self.size_l
        ifdat_corr_metadata['lines'] = self.size_y
        ifdat_corr_metadata['samples'] = self.size_x
        ifdat_corr_metadata['wavelength'] = self.meanwvl.astype(float).tolist()
        ifdat_corr_metadata['cat start time'] = self.start_time
        ifdat_corr_metadata['cat sclk start'] = self.sc_clock
        ifdat_corr_metadata['cat crism obsid'] = self.file_name[3:11]
        ifdat_corr_metadata['cat obs type'] = self.obs_type
        ifdat_corr_metadata['cat crism detector id'] = self.channel
        ifdat_corr_metadata['cat product version'] = 3
        ifdat_corr_metadata['cat binning mode'] = self.binmode
        ifdat_corr_metadata['wavelength filter'] = self.wvlfilter
        ifdat_corr_metadata['cat crism detector temp'] = self.detector_temp
        ifdat_corr_metadata['cat crism bench temp'] = self.bench_temp
        ifdat_corr_metadata['cat crism housing temp'] = self.spechousing_temp
        ifdat_corr_metadata['cat solar longitude'] = self.ls
        ifdat_corr_metadata['cat wa wave file'] = 'aux_files/WA/'+self.data_wvl_file
        ifdat_corr_metadata['cat ir waves reversed'] = 'YES'
        ifdat_corr_metadata['cat photometric correction flag'] = -1
        ifdat_corr_metadata['cat atmospheric correction flag'] = -1
        try:
            output_filename = os.path.join(output_folder,str.lower(self.file_name)+'_CAT_corr.hdr')
            envi.save_image(output_filename, self.ifdat_corr, dtype='float32', interleave = 'bsq', metadata=ifdat_corr_metadata,  force=True)
        except:
            print(self.obs_name+" corr data cannot be save")
    def update_hdr_file(self,hdr_file_path):
        with open(hdr_file_path, 'r') as file:
            lines = file.readlines()
        band_names_start = None
        band_names_end = None
        for i, line in enumerate(lines):
            if line.strip().startswith('band names = {'):
                band_names_start = i
            if band_names_start is not None and '}' in line:
                band_names_end = i
                break
        if band_names_start is not None and band_names_end is not None:
            del lines[band_names_start:band_names_end+1]
        data_ignore_value_updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith('data ignore value'):
                lines[i] = "data ignore value = 6.55350000e+004\n"
                data_ignore_value_updated = True
                break
        new_lines = [line for line in lines if 'default bands' not in line]
        new_lines.append('wavelength units = Micrometers\n')
        new_lines.append("default bands = {233, 78, 13}\n")
        wavelength_line = "wavelength = {" + ', '.join(map(str, self.meanwvl)) + "}\n"
        new_lines.append(wavelength_line)
        with open(hdr_file_path, 'w') as file:
            file.writelines(new_lines)
        print(f"Header file has been updated: {hdr_file_path}")

            
    def wavelength2index(self, wavelength):
        """Returns the index of the band with the closest wavelength number."""
        indices = []
        size_x, size_l = self.wvlc.shape
        for i in range(size_x):    
            index = np.argmin(np.abs(self.wvlc[i] - wavelength))
            indices.append(index)
        return indices

    def image_get_band(self, wavelength):
        """Extracts the band closest to the specified wavelength."""
        size_y, size_x, size_l = self.ifdat_corr.shape
        indices = self.wavelength2index(wavelength)
        reflectance = np.zeros((size_y, size_x, 1))
        for i in range(size_x):
            index = int(indices[i])  
            reflectance[:, i, 0] = self.ifdat_corr[:, i, index].flatten()
        return reflectance

    def wlambda(self, wavelength):
        """Returns the actual wavelength values for a given wavelength across all pixels."""
        indices = self.wavelength2index(wavelength)
        size_x, size_l = self.wvlc.shape
        wlambdas = np.zeros((size_x, 1))
        for i in range(size_x):
            wlambdas[i, 0] = self.wvlc[i, indices[i]]
        return wlambdas
    # Spectral paramter Viviano et al., 2014 JGR-Planet
    def R1330(self):
        return self.image_get_band(1.33)
    def OLINDEX3(self):
        R1750 = self.image_get_band(1.75)
        R2400 = self.image_get_band(2.4)
        wlambda1750 = self.wlambda(1.75)
        wlambda2400 = self.wlambda(2.4)
        slope = (R2400 - R1750) / (wlambda2400 - wlambda1750)
        # Calculate reflectance at baseline and actual reflectance at each wavelength
        # Then calculate band ratio (RB) for each wavelength
        wavelengths = [1.08, 1.152, 1.21, 1.25, 1.263, 1.276, 1.33, 1.368, 1.395, 1.427, 1.47]
        weights = [0.03, 0.03, 0.03, 0.03, 0.07, 0.07, 0.12, 0.12, 0.14, 0.18, 0.18]
        RB_values = []
        for wl, weight in zip(wavelengths, weights):
            RC = R1750 + slope * (self.wlambda(wl) - wlambda1750)
            R = self.image_get_band(wl)
            RB = (RC - R) / RC
            RB_values.append(RB * weight)
        return sum(RB_values)
    def LCPINDEX2(self):
        R1560 = self.image_get_band(1.56)
        R2450 = self.image_get_band(2.45)
        wlambda1560 = self.wlambda(1.56)
        wlambda2450 = self.wlambda(2.45)
        slope = (R2450 - R1560) / (wlambda2450 - wlambda1560)

        RC1690 = R1560 + slope * (self.wlambda(1.69) - wlambda1560)
        R1690 = self.image_get_band(1.69)
        RB1690 = (RC1690 - R1690) / RC1690

        RC1750 = R1560 + slope * (self.wlambda(1.75) - wlambda1560)
        R1750 = self.image_get_band(1.75)
        RB1750 = (RC1750 - R1750) / RC1750

        RC1810 = R1560 + slope * (self.wlambda(1.81) - wlambda1560)
        R1810 = self.image_get_band(1.81)
        RB1810 = (RC1810 - R1810) / RC1810

        RC1870 = R1560 + slope * (self.wlambda(1.87) - wlambda1560)
        R1870 = self.image_get_band(1.87)
        RB1870 = (RC1870 - R1870) / RC1870

        return RB1690 * 0.2 + RB1750 * 0.2 + RB1810 * 0.3 + RB1870 * 0.3

    def HCPINDEX2(self):
        R1690 = self.image_get_band(1.69)
        R2530 = self.image_get_band(2.53)
        wlambda1690 = self.wlambda(1.69)
        wlambda2530 = self.wlambda(2.53)
        slope = (R2530 - R1690) / (wlambda2530 - wlambda1690)

        RC2120 = R1690 + slope * (self.wlambda(2.12) - wlambda1690)
        R2120 = self.image_get_band(2.12)
        RB2120 = (RC2120 - R2120) / RC2120

        RC2140 = R1690 + slope * (self.wlambda(2.14) - wlambda1690)
        R2140 = self.image_get_band(2.14)
        RB2140 = (RC2140 - R2140) / RC2140

        RC2230 = R1690 + slope * (self.wlambda(2.23) - wlambda1690)
        R2230 = self.image_get_band(2.23)
        RB2230 = (RC2230 - R2230) / RC2230

        RC2250 = R1690 + slope * (self.wlambda(2.25) - wlambda1690)
        R2250 = self.image_get_band(2.25)
        RB2250 = (RC2250 - R2250) / RC2250

        RC2430 = R1690 + slope * (self.wlambda(2.43) - wlambda1690)
        R2430 = self.image_get_band(2.43)
        RB2430 = (RC2430 - R2430) / RC2430

        RC2460 = R1690 + slope * (self.wlambda(2.46) - wlambda1690)
        R2460 = self.image_get_band(2.46)
        RB2460 = (RC2460 - R2460) / RC2460

        return RB2120 * 0.1 + RB2140 * 0.1 + RB2230 * 0.15 + RB2250 * 0.3 + RB2430 * 0.2 + RB2460 * 0.15

    def ISLOPE1(self):
        R1815 = self.image_get_band(1.815)
        R2530 = self.image_get_band(2.53)
        wlamdal = self.wlambda(2.53)
        wlamdas = self.wlambda(1.815)
        return (R2530 - R1815) / (wlamdal - wlamdas)
    def BD1300(self):
        R1320 = self.image_get_band(1.32)
        R1080 = self.image_get_band(1.08)
        R1750 = self.image_get_band(1.75)
        wlambdac = self.wlambda(1.32)
        wlambdas = self.wlambda(1.08)
        wlambdal = self.wlambda(1.75)
        b = (wlambdac-wlambdas)/(wlambdal-wlambdas)
        a = 1 -b
        return 1-(R1320/(a*R1080+b*R1750))
    def BD1400(self):
        R1395 = self.image_get_band(1.395)
        R1330 = self.image_get_band(1.33)
        R1467 = self.image_get_band(1.467)
        wlambdac = self.wlambda(1.395)
        wlambdas = self.wlambda(1.33)
        wlambda1 = self.wlambda(1.467)
        b = (wlambdac - wlambdas) / (wlambda1 - wlambdas)
        a = 1 - b
        return 1 - (R1395 / (a * R1330 + b * R1467))

    def BD1435(self):
        R1435 = self.image_get_band(1.435)
        R1370 = self.image_get_band(1.37)
        R1470 = self.image_get_band(1.47)
        wlambdac = self.wlambda(1.435)
        wlambdas = self.wlambda(1.37)
        wlambdal = self.wlambda(1.47)
        b = (wlambdac - wlambdas) / (wlambdal - wlambdas)
        a = 1 - b
        return 1 - (R1435 / (a * R1370 + b * R1470))

    def BD1500_2(self):
        R1525 = self.image_get_band(1.525)
        R1367 = self.image_get_band(1.367)
        R1808 = self.image_get_band(1.808)
        wlambdac = self.wlambda(1.525)
        wlambdas = self.wlambda(1.367)
        wlambdal = self.wlambda(1.808)
        b = (wlambdac - wlambdas) / (wlambdal - wlambdas)
        a = 1 - b
        return 1 - (R1525 / (a * R1367 + b * R1808))

    def ICER1_2(self):
        return 1 - ((1 - self.BD1435()) / (1 - self.BD1500_2()))

    def BD1750_2(self):
        R1750 = self.image_get_band(1.75)
        R1690 = self.image_get_band(1.69)
        R1815 = self.image_get_band(1.815)
        wlambdac = self.wlambda(1.75)
        wlambdas = self.wlambda(1.69)
        wlambdal = self.wlambda(1.815)
        b = (wlambdac - wlambdas) / (wlambdal - wlambdas)
        a = 1 - b
        return 1 - (R1750 / (a * R1690 + b * R1815))
    
    def BD1900_2(self):
        R1930 = self.image_get_band(1.93)
        R1850 = self.image_get_band(1.85)
        R2067 = self.image_get_band(2.067)
        R1985 = self.image_get_band(1.985)
        wlambdac1 = self.wlambda(1.93)
        wlambdac2 = self.wlambda(1.985)
        wlambdas = self.wlambda(1.85)
        wlambdal = self.wlambda(2.067)
        b1 = (wlambdac1 - wlambdas) / (wlambdal - wlambdas)
        a1 = 1 - b1
        b2 = (wlambdac2 - wlambdas) / (wlambdal - wlambdas)
        a2 = 1 - b2
        return 0.5 * (1 - (R1930 / (a1 * R1850 + b1 * R2067))) + 0.5 * (1 - (R1985 / (a2 * R1850 + b2 * R2067)))

    def BD1900r2(self):
        R1850 = self.image_get_band(1.85)
        R2060 = self.image_get_band(2.06)
        wlambda1850 = self.wlambda(1.85)
        wlambda2060 = self.wlambda(2.06)
        slope = (R2060 - R1850) / (wlambda2060 - wlambda1850)
        reflective_continuum = lambda wl: R1850 + slope * (self.wlambda(wl) - wlambda1850)

        def calc_ratio(wl, reflective_continuum):
            R = self.image_get_band(wl)
            RC = reflective_continuum(wl)
            return R / RC

        upper_wls = [1.908, 1.914, 1.921, 1.928, 1.934, 1.941]
        lower_wls = [1.862, 1.869, 1.875, 2.112, 2.12, 2.126]

        upper_sum = sum(calc_ratio(wl, reflective_continuum) for wl in upper_wls)
        lower_sum = sum(calc_ratio(wl, reflective_continuum) for wl in lower_wls)

        return 1 - upper_sum / lower_sum
    def BD2100_2(self):
        R2132 = self.image_get_band(2.132)
        R1930 = self.image_get_band(1.93)
        R2250 = self.image_get_band(2.25)
        wlambdac = self.wlambda(2.132)
        wlambdas = self.wlambda(1.93)
        wlambdal = self.wlambda(2.25)
        b = (wlambdac - wlambdas) / (wlambdal - wlambdas)
        a = 1 - b
        return 1 - (R2132 / (a * R1930 + b * R2250))

    def BD2165(self):
        R2165 = self.image_get_band(2.165)
        R2120 = self.image_get_band(2.12)
        R2230 = self.image_get_band(2.23)
        wlambdac = self.wlambda(2.165)
        wlambdas = self.wlambda(2.12)
        wlambdal = self.wlambda(2.23)
        b = (wlambdac - wlambdas) / (wlambdal - wlambdas)
        a = 1 - b
        return 1 - (R2165 / (a * R2120 + b * R2230))

    def BD2190(self):
        R2185 = self.image_get_band(2.185)
        R2120 = self.image_get_band(2.12)
        R2250 = self.image_get_band(2.25)
        wlambdac = self.wlambda(2.185)
        wlambdas = self.wlambda(2.12)
        wlambdal = self.wlambda(2.25)
        b = (wlambdac - wlambdas) / (wlambdal - wlambdas)
        a = 1 - b
        return 1 - (R2185 / (a * R2120 + b * R2250))

    def MIN2200(self):
        R2165 = self.image_get_band(2.165)
        R2120 = self.image_get_band(2.12)
        R2350 = self.image_get_band(2.35)
        wlambdac1 = self.wlambda(2.165)
        wlambdas = self.wlambda(2.12)
        wlambdal = self.wlambda(2.35)
        b1 = (wlambdac1 - wlambdas) / (wlambdal - wlambdas)
        a1 = 1 - b1
        M2165 = 1 - R2165 / (a1 * R2120 + b1 * R2350)

        R2210 = self.image_get_band(2.21)
        wlambdac2 = self.wlambda(2.21)
        b2 = (wlambdac2 - wlambdas) / (wlambdal - wlambdas)
        a2 = 1 - b2
        M2210 = 1 - R2210 / (a2 * R2120 + b2 * R2350)

        stack = np.stack((M2165,M2210),axis=3)
        return np.min(stack,axis=3)
    
    def BD2210(self):
        R2210 = self.image_get_band(2.21)
        R2165 = self.image_get_band(2.165)
        R2250 = self.image_get_band(2.25)
        wlambdac = self.wlambda(2.21)
        wlambdas = self.wlambda(2.165)
        wlambdal = self.wlambda(2.25)
        b = (wlambdac - wlambdas) / (wlambdal - wlambdas)
        a = 1 - b
        return 1 - (R2210 / (a * R2165 + b * R2250))

    def D2200(self):
        R1815 = self.image_get_band(1.815)
        R2430 = self.image_get_band(2.43)
        slope = (R2430 - R1815) / (self.wlambda(2.43) - self.wlambda(1.815))
        R2210 = self.image_get_band(2.21)
        R2230 = self.image_get_band(2.23)
        R2165 = self.image_get_band(2.165)

        RC2210 = R1815 + slope * (self.wlambda(2.21) - self.wlambda(1.815))
        RC2230 = R1815 + slope * (self.wlambda(2.23) - self.wlambda(1.815))
        RC2165 = R1815 + slope * (self.wlambda(2.165) - self.wlambda(1.815))

        return 1 - (R2210 / RC2210 + R2230 / RC2230) / (2 * R2165 / RC2165)

    def BD2230(self):
        R2235 = self.image_get_band(2.235)
        R2210 = self.image_get_band(2.21)
        R2252 = self.image_get_band(2.252)
        b = (self.wlambda(2.235) - self.wlambda(2.21)) / (self.wlambda(2.252) - self.wlambda(2.21))
        a = 1 - b
        return 1 - (R2235 / (a * R2210 + b * R2252))

    def BD2250(self):
        R2245 = self.image_get_band(2.245)
        R2120 = self.image_get_band(2.12)
        R2340 = self.image_get_band(2.34)
        b = (self.wlambda(2.245) - self.wlambda(2.12)) / (self.wlambda(2.34) - self.wlambda(2.12))
        a = 1 - b
        return 1 - (R2245 / (a * R2120 + b * R2340))

    def BD2265(self):
        R2265 = self.image_get_band(2.265)
        R2210 = self.image_get_band(2.21)
        R2340 = self.image_get_band(2.34)
        b = (self.wlambda(2.265) - self.wlambda(2.21)) / (self.wlambda(2.34) - self.wlambda(2.21))
        a = 1 - b
        return 1 - (R2265 / (a * R2210 + b * R2340))

    def BD2290(self):
        R2290 = self.image_get_band(2.29)
        R2250 = self.image_get_band(2.25)
        R2350 = self.image_get_band(2.35)
        b = (self.wlambda(2.29) - self.wlambda(2.25)) / (self.wlambda(2.35) - self.wlambda(2.25))
        a = 1 - b
        return 1 - (R2290 / (a * R2250 + b * R2350))
    def D2300(self):
        R1815 = self.image_get_band(1.815)
        R2430 = self.image_get_band(2.43)
        slope = (R2430 - R1815) / (self.wlambda(2.43) - self.wlambda(1.815))
        RC2290 = R1815 + slope * (self.wlambda(2.29) - self.wlambda(1.815))
        R2290 = self.image_get_band(2.29)
        R2320 = self.image_get_band(2.32)
        R2330 = self.image_get_band(2.33)
        R2120 = self.image_get_band(2.12)
        R2170 = self.image_get_band(2.17)
        R2210 = self.image_get_band(2.21)
        RC2320 = R1815 + slope * (self.wlambda(2.32) - self.wlambda(1.815))
        RC2330 = R1815 + slope * (self.wlambda(2.33) - self.wlambda(1.815))
        RC2120 = R1815 + slope * (self.wlambda(2.12) - self.wlambda(1.815))
        RC2170 = R1815 + slope * (self.wlambda(2.17) - self.wlambda(1.815))
        RC2210 = R1815 + slope * (self.wlambda(2.21) - self.wlambda(1.815))

        numerator = R2290 / RC2290 + R2320 / RC2320 + R2330 / RC2330
        denominator = (R2120 / RC2120 + R2170 / RC2170 + R2210 / RC2210) / 3
        return 1 - numerator / denominator

    def BD2355(self):
        R2355 = self.image_get_band(2.355)
        R2300 = self.image_get_band(2.3)
        R2450 = self.image_get_band(2.45)
        b = (self.wlambda(2.355) - self.wlambda(2.3)) / (self.wlambda(2.45) - self.wlambda(2.3))
        a = 1 - b
        return 1 - (R2355 / (a * R2300 + b * R2450))

    def SINDEX2(self):
        R2120 = self.image_get_band(2.12)
        R2400 = self.image_get_band(2.4)
        R2290 = self.image_get_band(2.29)
        return 1 - ((R2120 + R2400) / 2 * R2290)

    def ICER2_2(self):
        R2456 = self.image_get_band(2.456)
        R2530 = self.image_get_band(2.53)
        slope = (R2456 - R2530) / (self.wlambda(2.456) - self.wlambda(2.53))
        RC2600 = R2456 + slope * (self.wlambda(2.6) - self.wlambda(2.456))
        return RC2600
    def MIN2295_2480(self):
        R2295 = self.image_get_band(2.295)
        R2165 = self.image_get_band(2.165)
        R2364 = self.image_get_band(2.364)
        M2295 = 1 - R2295 / ((self.wlambda(2.295) - self.wlambda(2.165)) / (self.wlambda(2.364) - self.wlambda(2.165)) * (R2165 - R2364) + R2364)
        
        R2480 = self.image_get_band(2.48)
        R2365 = self.image_get_band(2.365)
        R2570 = self.image_get_band(2.57)
        M2480 = 1 - R2480 / ((self.wlambda(2.48) - self.wlambda(2.365)) / (self.wlambda(2.57) - self.wlambda(2.365)) * (R2365 - R2570) + R2570)
        stack = np.stack((M2295,M2480),axis=3)
        return np.min(stack,axis = 3)
    def MIN2345_2537(self):
        R2345 = self.image_get_band(2.345)
        R2250 = self.image_get_band(2.25)
        R2430 = self.image_get_band(2.43)
        M2345 = 1 - R2345 / ((self.wlambda(2.345) - self.wlambda(2.25)) / (self.wlambda(2.43) - self.wlambda(2.25)) * (R2250 - R2430) + R2430)
        
        R2537 = self.image_get_band(2.537)
        R2602 = self.image_get_band(2.602)
        M2537 = 1 - R2537 / ((self.wlambda(2.537) - self.wlambda(2.43)) / (self.wlambda(2.602) - self.wlambda(2.43)) * (R2430 - R2602) + R2602)
        stack = np.stack((M2345,M2537),axis =3)
        return np.min(stack,axis = 3)
    def BD2500_2(self):
        R2480 = self.image_get_band(2.48)
        R2570 = self.image_get_band(2.57)
        R2364 = self.image_get_band(2.364)
        M2500_2 = 1 - R2480 / ((self.wlambda(2.48) - self.wlambda(2.364)) / (self.wlambda(2.57) - self.wlambda(2.364)) * (R2364 - R2570) + R2570)
        return M2500_2
    def BD3000(self):
        R3000 = self.image_get_band(3)
        R2530 = self.image_get_band(2.53)
        R2210 = self.image_get_band(2.21)
        return 1 - (R3000 / (R2530 * (R2530 / R2210)))

    def BD3100(self):
        R3120 = self.image_get_band(3.12)
        R3000 = self.image_get_band(3)
        R3250 = self.image_get_band(3.25)
        b = (self.wlambda(3.12) - self.wlambda(3)) / (self.wlambda(3.25) - self.wlambda(3))
        a = 1 - b
        return 1 - (R3120 / (a * R3000 + b * R3250))

    def BD3200(self):
        R3320 = self.image_get_band(3.32)
        R3250 = self.image_get_band(3.25)
        R3390 = self.image_get_band(3.39)
        b = (self.wlambda(3.32) - self.wlambda(3.25)) / (self.wlambda(3.39) - self.wlambda(3.25))
        a = 1 - b
        return 1 - (R3320 / (a * R3250 + b * R3390))

    def BD3400_2(self):
        R3420 = self.image_get_band(3.42)
        R3250 = self.image_get_band(3.25)
        R3630 = self.image_get_band(3.63)
        b = (self.wlambda(3.42) - self.wlambda(3.25)) / (self.wlambda(3.63) - self.wlambda(3.25))
        a = 1 - b
        return 1 - (R3420 / (a * R3250 + b * R3630))

    def CINDEX2(self):
        R3450 = self.image_get_band(3.45)
        R3875 = self.image_get_band(3.875)
        R3610 = self.image_get_band(3.61)
        b = (self.wlambda(3.61) - self.wlambda(3.45)) / (self.wlambda(3.875) - self.wlambda(3.45))
        a = 1 - b
        return 1 - ((a * R3450 + b * R3875) / R3610)
    def R440(self):
        return self.image_get_band(0.44)
    def R550(self):
        return self.image_get_band(0.55)
    def R600(self):
        return self.image_get_band(0.6)
    def IRR1(self):
        R800 = self.image_get_band(0.8)
        R997 = self.image_get_band(0.997)
        return R800/R997
    def R1080(self):
        return self.image_get_band(1.08)
    def R1506(self):
        return self.image_get_band(1.506)
    def R2529(self):
        return self.image_get_band(2.529)
    def BD2600(self):
        R2600 = self.image_get_band(2.6)
        R2530 = self.image_get_band(2.53)
        R2630 = self.image_get_band(2.63)
        wlambdac = self.wlambda(2.6)
        wlambdas = self.wlambda(2.53)
        wlambdal = self.wlambda(2.63)
        b = (wlambdac-wlambdas)/(wlambdal-wlambdas)
        a = 1 - b
        return 1-(R2600/a*R2530+b*R2630)
    def IRR2(self):
        R2530 = self.image_get_band(2.53)
        R2210 = self.image_get_band(2.21)
        return R2530/R2210
    def IRR3(self):
        R3500 = self.image_get_band(3.5)
        R3390 = self.image_get_band(3.39)
        return R3500/R3390
    def R3920(self):
        return self.image_get_band(3.92)
    def summaryproduct(self):
        # Define a list of method references, not their call results
        product_methods = [
            self.R1330, self.BD1300, self.OLINDEX3, self.LCPINDEX2, self.HCPINDEX2,
            self.ISLOPE1, self.BD1400, self.BD1435, self.BD1500_2, self.ICER1_2,
            # Add other method references as needed
            self.BD1750_2, self.BD1900_2, self.BD1900r2, self.BD2100_2, self.BD2165,
            self.BD2190, self.MIN2200, self.BD2210, self.D2200, self.BD2230,
            self.BD2250, self.BD2265, self.BD2290, self.D2300, self.BD2355,
            self.SINDEX2, self.ICER2_2, self.MIN2295_2480, self.MIN2345_2537,
            self.BD2500_2, self.BD3000, self.BD3100, self.BD3200, self.BD3400_2,
            self.CINDEX2, self.R2529, self.BD2600, self.IRR2, self.IRR3, self.R3920
        ]

        # Call each method dynamically and calculate the products
        products = [method() for method in product_methods]

        # Stack all products to create a multi-layered array
        self.summary_product = np.dstack(products)

    def save_summaryproduct(self,output_folder = None):
        if output_folder is None:
            output_folder = self.input_folder
        product_names = [
            'R1330', 'BD1300', 'OLINDEX3', 'LCPINDEX2', 'HCPINDEX2', 'ISLOPE1',
            'BD1400', 'BD1435', 'BD1500_2', 'ICER1_2', 'BD1750_2', 'BD1900_2',
            'BD1900r2', 'BD2100_2', 'BD2165', 'BD2190', 'MIN2200', 'BD2210',
            'D2200', 'BD2230', 'BD2250', 'BD2265', 'BD2290', 'D2300', 'BD2355',
            'SINDEX2', 'ICER2_2', 'MIN2295_2480', 'MIN2345_2537', 'BD2500_2',
            'BD3000', 'BD3100', 'BD3200', 'BD3400_2', 'CINDEX2', 'R2529', 'BD2600', 'IRR2', 'IRR3', 'R3920'
        ]
        metadata = {
            'lines': self.size_y,
            'samples': self.size_x,
            'bands': self.summary_product.shape[2],
            'data type': 4,
            'band names': product_names  # Include the list of band names in the metadata
        }
        output_path = os.path.join(output_folder, str.lower(self.file_name) + '_CAT_corr_summaryproduct.hdr')
        
        envi.save_image(output_path, self.summary_product, metadata=metadata, force=True, interleave='bil')
    def project_summaryproduct(self):
        if self.summary_product is None:
            print("No summary product available. Calculating now.")
            self.summaryproduct()
        ifdat = np.array(self.summary_product)
        geocube = self.geomdata  # Assuming this is where your lat/lon data is stored
        latlon = geocube[:, :, 3:5]
        dlatlon = np.array([latlon.shape[1], latlon.shape[0]])
        centerlon = latlon[dlatlon[1] // 2, dlatlon[0] // 2, 1]
        ctrl_pts_file = os.path.join(self.input_folder, "ctrl_pts.txt")

        with open(ctrl_pts_file, "w") as file:
            for i in range(5, dlatlon[1] + 1, dlatlon[1] // 10):
                for j in range(5, dlatlon[0] + 1, dlatlon[0] // 10):
                    if i > dlatlon[1]:
                        i = dlatlon[1]
                    if j > dlatlon[0]:
                        j = dlatlon[0]
                    file.write(f"-gcp {j} {i} {latlon[i-1, j-1, 1]} {latlon[i-1, j-1, 0]}\n")

        rows, cols, bands = ifdat.shape
        driver = gdal.GetDriverByName('GTiff')
        temp_image = os.path.join(self.input_folder, "project_crism_temp_image.tif")
        dataset = driver.Create(temp_image, cols, rows, bands, gdal.GDT_Float32)

        for i in range(bands):
            band = dataset.GetRasterBand(i+1)
            band.WriteArray(ifdat[:, :, i])
            dataset.FlushCache()

        translated_image = os.path.join(self.input_folder, "project_crism_image.tif1")
        gdal_translate_cmd = f'gdal_translate -of GTIFF -a_srs "+proj=longlat +a=3396190 +b=3376200 +units=m +no_defs" `cat {ctrl_pts_file}` "{temp_image}" "{translated_image}"'
        subprocess.run(gdal_translate_cmd, shell=True, check=True)

        projected_image = os.path.join(self.input_folder, "project_crism_image.tif2")
        gdalwarp_cmd = (
            f'gdalwarp -of GTIFF -q -rb -srcnodata 0 '
            f'-t_srs "+proj=sinu +lon_0={centerlon} +x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs" '
            f'-tps "{translated_image}" "{projected_image}"'
        )
        subprocess.run(gdalwarp_cmd, shell=True)

        final_output = os.path.join(self.input_folder, str.lower(self.file_name)+'_CAT_corr_summaryproduct_p.tif')
        os.rename(projected_image, final_output)
        os.remove(ctrl_pts_file)
        os.remove(temp_image)
        os.remove(translated_image)
        self.convert_tif_to_envi(final_output,final_output[:-4])
        product_names = [
            'R1330', 'BD1300', 'OLINDEX3', 'LCPINDEX2', 'HCPINDEX2', 'ISLOPE1',
            'BD1400', 'BD1435', 'BD1500_2', 'ICER1_2', 'BD1750_2', 'BD1900_2',
            'BD1900r2', 'BD2100_2', 'BD2165', 'BD2190', 'MIN2200', 'BD2210',
            'D2200', 'BD2230', 'BD2250', 'BD2265', 'BD2290', 'D2300', 'BD2355',
            'SINDEX2', 'ICER2_2', 'MIN2295_2480', 'MIN2345_2537', 'BD2500_2',
            'BD3000', 'BD3100', 'BD3200', 'BD3400_2', 'CINDEX2', 'R2529', 'BD2600', 'IRR2', 'IRR3', 'R3920'
        ]
        self.update_summary_product_hdr(final_output[:-4]+'.hdr',product_names)
        os.remove(final_output)
    def update_summary_product_hdr(self,hdr_file_path, summary_product_names):
        with open(hdr_file_path, 'r') as file:
            lines = file.readlines()
        # Remove existing "band names" entry
        new_lines = []
        inside_band_names = False
        for line in lines:
            if line.strip().startswith('band names = {'):
                inside_band_names = True
            elif inside_band_names and '}' in line:
                inside_band_names = False
                continue  # Skip this line to remove the closing of band names
            elif not inside_band_names:
                new_lines.append(line)
        # Add new "band names" entry
        band_names_str = 'band names = {' + ', '.join(summary_product_names) + '}\n'
        new_lines.insert(-1, band_names_str)
        with open(hdr_file_path, 'w') as file:
            file.writelines(new_lines)
