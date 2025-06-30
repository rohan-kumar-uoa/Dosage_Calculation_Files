import numpy as np
import matplotlib.pyplot as plt
import spekpy as sp
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from numpy import array
import requests
import bs4

from functools import reduce

def populate_element_parameters(recalc = False):
    """ Extracts a dataframe storing energy, mass attenuation coefficient, and mass absorption coefficient data for each element.
        If this has already been done, returns the dataframe, otherwise it is calculated.
        Parameters
        __________
        recalc : bool
            Set true if this is the first time running this library locally to create the .csv file

        Returns
        _______
        df : pd.DataFrame
            Dataframe storing elemental properties 
    """
    if recalc:
        arr = get_elementlist2()
        nist_baseURL = 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z@.html'
        element_dataframe = pd.DataFrame(columns= ['E', 'murho', 'muenrho', 'element', 'elnum'])
        for elname, elnum in zip(arr['Symbol'], arr['Atomic Number']):
            if len(str(elnum)) < 2:
                elnumstring = "0" + str(elnum)
            else:
                elnumstring = str(elnum)

            element_URL = nist_baseURL.replace('@', elnumstring)

            req = requests.get(element_URL)
            soup = bs4.BeautifulSoup(req.text, 'html.parser')

            table = soup.find_all('pre')[0]
            mystring = table.text.strip()

            new = mystring.split('\n')[6:]

            if new[0] == '':
                new = new[1:]

            if elnum > 10:
                new = [new[i][3:] for i in range(len(new))]
            
            new = array([el[0:-1].split('  ') for el in new])

            new = pd.DataFrame(new, columns= ['E', 'murho', 'muenrho'])
            new['element'] = elname
            new['elnum'] = str(elnum)

            element_dataframe = pd.concat([element_dataframe, new])
            print(elname,elnum, " Done")
        
        addtwoonend = lambda s: (s + '2') if s[-2] == '-' else s
        spacereplace = lambda s : s.replace(' ','')

        element_dataframe['muenrho'] = element_dataframe['muenrho'].apply(addtwoonend)

        for colname, coldata in element_dataframe.items():
            element_dataframe[colname] = element_dataframe[colname].apply(spacereplace)

        element_dataframe['E'] = element_dataframe['E'].astype(float)
        element_dataframe['murho'] = element_dataframe['murho'].astype(float)
        element_dataframe['muenrho'] = element_dataframe['muenrho'].astype(float)
        element_dataframe['elnum'] = element_dataframe['elnum'].astype(int)
        element_dataframe.to_csv('Element_data_master.csv')

        return element_dataframe
    else:
        return pd.read_csv("Data Files/Element_data_master.csv")

def normalize_dict(d):
    """Takes in a dictionary with all values numeric. Scales all values so they sum to 1"""
    total = sum(d.values())
    return dict((k, v/total) for k,v in d.items())

def find_coefficient(energy, fluence, weights = None):
    """ Calculates the mass absorption coefficient for a given Xray spectrum and compound.

        Parameters
        __________
        energy : array
            Energy Bins
        fluence : array
            Fluences in each energy bin
        weights : dict{'str' : float}
            Dictionary storing compounds and their respective weight fractions

        Returns
        _______
        muenrho : float
            Mass absorption coeffient 
    """
    
    elementdf = populate_element_parameters()

    test = pd.read_csv(r"Data Files/borosilicate.csv", header = None, names = ['E', 'murho', 'muenrho'])
    test['element'] = 'borosilicate'
    test['elnum'] = np.float64('NaN')
    elementdf = pd.concat([elementdf, test])

    sum = np.zeros(len(energy))
    pdf = fluence/np.trapezoid(fluence,energy)
    els = weights.keys()
    masterplot = False
    for elname in els:
        edf = elementdf[elementdf['element'] == elname]
        en = edf['E']*1e3
        murho = edf['muenrho']

        
        interp_vals = np.exp(np.interp(np.log(energy), np.log(en), np.log(murho)))

        sum += weights[elname]*interp_vals

    contrib = sum*pdf
    mu_en_rho = float(np.trapezoid(contrib, energy ))
    contrib /= mu_en_rho
    if masterplot:
        fig,ax1 = plt.subplots()
        ax1.plot(energy, contrib, label = 'contributions')
        # ax1.legend(title = r"$\frac{\mu_{en}}{\rho}$ (${cm}^2/g$)")
        # ax1.legend()
        ax1.set_xlabel("Energy (keV)")
        ax1.set_ylabel("Normalized Contributions" + r"[${cm}^2/g/keV$]")
        ax1.set_title(r"Contributions to $\frac{\mu_{en}}{\rho}$ for superconductor "+"\n(area under curve normalized to 1)")
        #  + f"\n{mu_en_rho:.3f}" + r"${cm}^2/g$"
        # plt.xlim([0,100])

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Cumulative integral")
        cumul_int = cumulative_trapezoid(contrib, energy)
        ax2.plot(energy[1:], cumul_int, 'r', linewidth = 2)
        # print(cumul_int[(energy >= 122.)[1:]])
        # print(energy)
        ax2.tick_params(axis='y', colors='red')
        ax2.yaxis.label.set_color('red')
        # ax2.legend()
        xmin,xmax,ymin,ymax = ax2.axis()

        # ax2.vlines([122], ymin = ymin, ymax = .05)
        cont_at_122 = cumul_int[(energy == 122.)[1:]][0]
        print(cont_at_122)
        ax2.annotate(f"{cont_at_122*100:.2f}% under 122keV" + f"\n{(1-cont_at_122)*100:.2f}% above 122keV", xy = (122.,cont_at_122), xytext = (140, cont_at_122-.1), arrowprops=dict(facecolor='black', shrink=0.01))
    print(f"{mu_en_rho}")
    
    return mu_en_rho

def get_elementlist2(recalc = False):
    """ Tabulates the symbol, atomic mass, and atomic number for the elements from an online resource.
        Stores the data in a dataframe and saves it locally to a csv

        Parameters
        __________
        recalc : bool
            Set true if this is the first time running this library locally to create the .csv file

        Returns
        _______
        df : pd.DataFrame
            Dataframe storing elemental properties 
    """

    if recalc:
        URL = 'https://byjus.com/chemistry/periodic-table-elements/'
        req = requests.get(URL)
        soup = bs4.BeautifulSoup(req.text, 'html.parser')

        str_remove = lambda s,substr: s.replace(f"<{substr}>",'').replace(f"</{substr}>",'')

        colnames = soup.find_all('tbody')[1]
        colnames = colnames.find_all('th')
        colnames = array([str_remove(str_remove(str(colnames[i]), 'th'), 'strong') for i in range(len(colnames))])

        body = soup.find_all('tbody')[1]
        body = body.find_all('td')
        body = array([str_remove(str(body[i]), 'td') for i in range(len(body))])
        ncols = 6
        body = np.reshape(body, (int(len(body)/ncols),ncols))

        df = pd.DataFrame(body, columns = colnames)[['Atomic Mass', 'Atomic Number', 'Symbol']]
        
        df['Atomic Number'] = df['Atomic Number'].astype(int)
        df['Atomic Mass'] = df['Atomic Mass'].astype(float)
        df['Symbol'] = df['Symbol'].astype(str)

        df = df[df['Atomic Number'] <= 92]
        df.to_csv('Element_Info.csv',index = False )
        return df
    else:
        df = pd.read_csv('Data Files/Element_Info.csv')
        return df
    
def mergedicts(dictarr):
    """ Merges an array of dictionaries. If there are common keys, their values are summed. 

        Parameters
        __________
        dictarr : array[dict]
            Array of dictionaries

        Returns
        _______
        merged : dict
            Merged Dictionary 
    """
    merged = {}
    all_keys = reduce(np.union1d, tuple([list(d.keys()) for d in dictarr]))

    for k in all_keys:
        merged[k] = 0
        for d in dictarr:
            for key,val in d.items():
                if k == key:
                    merged[k] += val

    return merged

def massfrac_to_molarfrac(massfrac_dict):
    """ Takes a dictionary storing compounds as keys and mass fractions as values.
        Converts the mass fractions to molar fractions 
    """
    el_df = get_elementlist2()
    n_els = {}
    for el, massfrac in massfrac_dict.items():
        M_el = float(el_df[el_df['Symbol'] == el]['Atomic Mass'].iloc[0])
        # print(M_el)
        n_els[el] = massfrac/M_el
    return n_els

def molarfrac_to_massfrac(molarfrac):
    """ Takes a dictionary storing compounds as keys and molar fractions as values.
        Converts the molar fractions to mass fractions 
    """
    composition = molarfrac
    el_df = get_elementlist2()
    comp_els = composition.keys()
    composition_M = array([float(el_df[el_df['Symbol'] == el]['Atomic Mass'].iloc[0]) for el in comp_els])
    composition_M = dict(zip(comp_els, composition_M))
    tot_weight = np.sum([composition[el]*composition_M[el] for el in comp_els])  
    weights = [composition[el]*composition_M[el]/tot_weight for el in comp_els]
    weights = dict(zip(comp_els, weights))
    return weights

def intensity_convert(keVcm2):
    """Converts an intensity from units of keV/cm^2 s to W/m^2"""
    eflu = float(keVcm2)
    eflu*= 1.602e-16    # J/cm^2 s   
    eflu*= (1e2)**2     # W/m^2
    return eflu

def print_floatdict(floatdict, title=''):
    """ Prints a dictionary of molar or mass fractions neatly"""
    print(title)
    for k,v in floatdict.items():
        print(f'{k:3}: {round(v,4)}')
    print('____________')

def eldict_to_formula(comp):
    """Extracts a molecular formula from the keys of a dictionary"""
    arr = []
    for k,v in comp.items():
        arr.append(k)
        if v >1:
            arr.append(str(v))
    return ''.join(arr)

def distribute_makeup(composition):
    """ Applies a weight to mass/molar fractions for an array of compounds

        Parameters
        __________
        composition : array[[dict,float]]
            Array of compounds, stored as dictionaries, and weighting percentages

        Returns
        _______
        composition : array[[dict,float]]
            Array of weighted compounds 

    """
    for compound, perc in composition:
        for k,v in compound.items():
            compound[k] = v * perc
    return composition

# Store composition of borosilicate glass by mass fraction
borosilicate_comp = {'B':4, 'O':54, 'Na':2.85, 'Al':1.1, 'Si':37.7, 'K':0.3}

# Store default sp.Spek keyword arguments. These are used unless overwritten by user specified arguments
default_spectrum = {'kvp':320., 'th':30., 'physics' : 'Spekcalc', 'targ' : 'W', 'dk' : 1, 'mas' : 10}

def calculate_absorbed_dosage(my_filters,composition, distance = 135.47e-1, dopant_type = 'molar', set_spectrum_params = {}):
    """	Calculates absorbed dosages from an Xray source for a compound with a specified composition.
        Prints the intensity at the target(W/m^2), the mass absorption coefficient (cm^2/g), and the absorbed dose (Gy/s)

    Parameters
    __________
    my_filters : array[tuple]
        Array of attenuating media, stored as (material, thickness (mm)) tuples. For a full list of supported materials, use sp.Spek.show_matls() from the Spekpy library. 
        For more information, see https://bitbucket.org/spekpy/spekpy_release/wiki/Function%20glossary#markdown-header-show_matls. Spekpy also allows custom materials to be defined.
    composition : array[[dict, float]]
        Composition of the target, stored as a list of compound molecular formulae, and molar/mass fraction pairs
        If a constitutent is borosilicate glass, pass 'borosilicate' instead of a dictionary for that pair.
    distance : float
        Distance from the source to the target in cm
    dopant_type : str
        Specifies whether the given percentages were mass fractions or molar fractions. Default is molar fractions
    set_spectrum_params: dict
        Overwrite the arguments of the sp.Spek function to alter the characteristics of the spectrum. The default arguments specify:
        {   kvp     = Voltage (kV)      = 320 kV,
            th      = Target Angle      = 30 degrees,
            physics = Simulation method = Spekcalc,
            targ    = Source material   = W,
            dk      = Energy Bin size   = 1 keV,
            mas     = MilliAmp-Seconds  = 10mA * 1s (1s for dosage rate in Gy/s)
        }
        For information on the supported keyword arguments, see: https://bitbucket.org/spekpy/spekpy_release/wiki/Further%20information#markdown-header-arguments-of-the-spek-class
    Returns
    _______
    target_params : tuple
        (intensity at target, mass absorption coefficient, absorbed dose)

    Notes
    _____

    __________
    EXAMPLE:
    />>> # 80% SiO2, 10% GeO2, 10% Er2O3
    />>> composition = [[{'Si':1, 'O':2}, 0.8], [{'Ge':1, 'O':2}, 0.1], [{'Er':2, 'O':3}, 0.1]]
    />>> # attenuation from 3mm of Be and 100 mm of Air 
    />>> my_filters=[('Be',3.),('Air', 100.),] 
    />>> # Specify weightings as molar fractions. Calculate the absorbed dose 103cm from the target
    />>> dose = calculate_absorbed_dosage(my_filters,composition, distance = 103., dopant_type = 'molar')
    Weight Fraction Composition
    Er : 0.2463
    Ge : 0.0891
    O  : 0.3888
    Si : 0.2757
    ____________
    Intensity at Target = 2.393022710935311 W/m^2 

    Mass Absorption Coefficient is 4.967986763196182cm^2/g

    Dose = 1.18885 Gy/s
    """
    # Check if all percentages add to 1:
    assert sum([perc for _,perc in composition]) == 1
       
    # Normalize the molar ratio of all compositions:
    for i in range(len(composition)):
        if composition[i][0] == 'borosilicate':
            borosilicate_masscomp = massfrac_to_molarfrac(borosilicate_comp)
            composition[i][0] = borosilicate_masscomp
        composition[i][0] = normalize_dict(composition[i][0])
    
    # If the compounds are given in molar ratios, weight the elements in each compound appropriately.
    if dopant_type == 'molar':
        composition = distribute_makeup(composition)
        # combine the dictionaries of all compounds:
        compounds = [comp for comp,perc in composition]
        final_composition = mergedicts(compounds)

        # convert to weight fractions
        weights = molarfrac_to_massfrac(final_composition)
    # Otherwise we have mass fractions
    elif dopant_type == 'mass':
        # Each of the compounds are specified in molar ratios, convert to mass ratios
        for i in range(len(composition)):
            composition[i][0] = molarfrac_to_massfrac(composition[i][0])

        # Apply weighting
        composition = distribute_makeup(composition)

        # Convert to weight fractions
        compounds = [comp for comp,perc in composition]
        weights = mergedicts(compounds)

    # print the weight fractions to the user.
    print_floatdict(weights, "Weight Fraction Composition")

    # weights = {'borosilicate':1}
    # print("USING BOROSILICATE OVERRIDE")

    # Set up characteristics of the spectrum, using Voltage=320kV, Target angle (from X-ray source datasheet) = 30degrees, Tungsten target, 
    # Voltage,kvp       = 320kV
    # Target Angle, th  = 30degrees
    # Target Type targ  = W
    # Energy Bins, dk   = 1keV
    # Distance, dz      = 135.47e-1 cm
    # Milli-Amp Seconds,mas = 10mAs (for dosage rate every second)
    
    # Update the default spectrum parameters with any new ones
    spectrum_params = default_spectrum | set_spectrum_params
    s = sp.Spek(**spectrum_params,z=distance)
    # Define filters using ('material', thickness) pairs
    s.multi_filter(my_filters)
    # Obtain the spectrum and calculate the expected mass-absorption coefficient
    # s.summarize()
    k,phi_k = s.get_spectrum(edges=True)
    
    df = pd.DataFrame({'energy (keV)':k, 'flux (keV^-1 cm^-2 s^-1)':phi_k})
    df.to_csv("spectrum_rawdata.csv", index = False)

    flux = s.get_flu()
    print(f"Photon Flux at Target = {flux:.5e} photons/(cm^2 s) \n")

    
    # Extract energy fluence and convert into dosage rate
    intens = intensity_convert(s.get_eflu())
    print(f"Intensity at Target = {intens:.5f} W/m^2 \n")

    plt.figure()
    plt.plot(k,phi_k)
    plt.xlabel("Energy (keV)")
    plt.ylabel(r"Photon Fluence $\left(\text{keV}^{-1}\text{cm}^{-2}\text{s}^{-1}\right) $")
    plt.title("Generated Spekpy Spectrum")
    plt.show()

    muenrho = find_coefficient(k, phi_k, weights = weights)
    print(f"Mass Absorption Coefficient is {muenrho:.5f}cm^2/g\n")
    dose = intens * (muenrho/10)         #get dosage rate
    print(f"Dosage Rate = {dose:.5f} Gy/s")

    return (intens, muenrho, dose)