import pandas as pd
import scipy.interpolate as sc
import numpy as np
import os
from bluesky import settings

settings.set_variable_defaults(perf_data_path_iLP = 'bluesky/resources/performance/iLP')


class EEI:
    def __init__(self):
        '''
        Definition: Return data-set of the highest degree of accuracy, and return elements of this data to the Traffic
                    object;
        Methods:
            select(): Given inputs of type, airline and/or DESTination, return data-set with highest degree of accuracy;
            TXXf(): Subfunction of select(), where the data-set is obtained from the data-set and interpolated;
            IAS_ROC(): Function to return IAS/CAS and ROC to the Traffic object;
            values(): Subfunction of IAS_ROC(), which computes the IAS/CAS and ROC for every aircraft agent;

        Created by: Lars Dijkstra
        Date: 14-10-2022

        Edited by: Arham Elahi
        Date: August 2024

        Edited by: Albert Chou
        Date: June 2025
        '''
        file = 'iLP_TOperf_2023_SID.csv'

        try:
            file = os.path.join(settings.perf_data_path_iLP, file)
        except:
            raise FileNotFoundError("Error, could not find file {}.".format(file))

        data = pd.read_csv(file, delimiter='\t', decimal=".", header=0, dtype = {'AC_id': str, 'data_type': str,
                                                                                  'ALT': float, 'IAS': float,
                                                                                  'MACH': float, 'ROCD': float,
                                                                                  'COLalt': float, 'airline': str,
                                                                                  'DEST': str, 'country': str,
                                                                                   'continent': str, 'num_ac': float,
                                                                                   'IAS_m': float, 'MACH_m': float,
                                                                                   'ROCD_m': float, 'IAS_sd': float,
                                                                                   'MACH_sd': float, 'ROCD_sd': float,
                                                                                   'min_calt': float, 'max_calt': float,
                                                                                   'mode_calat':float})
        # Grouped by defined degrees of accuracy
        data = data.groupby('data_type')

        # Store all degrees of accuracy in data types in different variables
        self.TAD = data.get_group("TAD").reset_index(drop=True)
        self.TADL = data.get_group("TADL").reset_index(drop=True)
        self.TADC = data.get_group("TADC").reset_index(drop=True)
        self.TA = data.get_group("TA").reset_index(drop=True)
        self.TD = data.get_group("TD").reset_index(drop=True)
        self.TCOU = data.get_group("TCOU").reset_index(drop=True)
        self.TCON = data.get_group("TCON").reset_index(drop=True)
        self.TT = data.get_group("TT").reset_index(drop=True)

        # Group dataframes by their respective parameters
        self.TADG = self.TAD.groupby(["AC_id", "airline", "DEST"])
        self.TADLG = self.TADL.groupby(["AC_id", "airline", "country"])
        self.TADCG = self.TADC.groupby(["AC_id", "airline", "continent"])
        self.TAG = self.TA.groupby(["AC_id", "airline"])
        self.TDG = self.TD.groupby(["AC_id", "DEST"])
        self.TCOUG = self.TCOU.groupby(["AC_id", "country"])
        self.TCONG = self.TCON.groupby(["AC_id", "continent"])
        self.TTG = self.TT.groupby(["AC_id"])

        # Obtain the keys from the grouped dataframes. E.g. for type, airline and destination this could be (A320, EZY,
        # EGGW)
        self.ac_airline_dest = self.TADG.groups.keys()
        self.ac_airline_country = self.TADLG.groups.keys()
        self.ac_airline_continent = self.TADCG.groups.keys()
        self.ac_airline = self.TAG.groups.keys()
        self.ac_dest = self.TDG.groups.keys()
        self.ac_country = self.TCOUG.groups.keys()
        self.ac_continent = self.TCONG.groups.keys()
        self.AC_ids = self.TT['AC_id'].unique()

        # We require the location of the airports to find the respective countries they are located in.
        self.airports = pd.read_csv(os.path.join(settings.perf_data_path_iLP, "iLP_AIRPORTS.csv"), sep=',', usecols=[1, 8])

        # We require to know which countries are located in which continent
        self.continents = pd.read_csv(os.path.join(settings.perf_data_path_iLP,'iLP_COUNTRIES.txt'), sep='\t')
        self.continents["two_code"] = self.continents["two_code"].apply(lambda x: x if not pd.isnull(x) else "NA")

        # Type of interpolation
        self.kind = "linear"

        # Constants
        self.kts = 0.514444
        self.fpm = 0.3048 / 60
        self.ft = 0.3048

    def select(self, ID, AC_type, SID, **kwargs):
        """
        Function: Returns the dataframe with the highest degree of accuracy to the Traffic object;

        Args:
            ID:         Callsign
            AC_type:    Aircraft id. Combination of ICAO code (e.g. "A320") and SID (e.g. "ARN4S")
            SID:        SID
            airline:    Airline code. Use ICAO code (e.g. "KLM")
            DEST:       DESTination airport. Use ICAO code (e.g. "EDDW")

        Returns:
            TXX_df:     Dataframe with IAS/CAS and ROC for aircraft agent
            IAS_slope_max: maximum slope of IAS per flight level from the dataframe

        Created by: Lars Dijkstra
        Date: 14-10-2022

        Edited by: Arham Elahi
        Date: July 2024
        """

        if "DEST" in kwargs:
            country = self.airports.loc[self.airports['ident'] == kwargs['DEST']].reset_index()

            if len(country) >0:
                country = country.at[0,"iso_country"]
                continent = self.continents.loc[self.continents['two_code'] == country].reset_index().at[0,'continent']
            else:
                country = None
                continent = None

        swA = False

        # Check for the best possible database for AC_type with SID
        if SID is not None and SID != "NULLSID":
            AC_id = AC_type + '_' + SID

            if "DEST" in kwargs:
                if "airline" in kwargs:
                    if (AC_id, kwargs["airline"], kwargs["DEST"]) in self.ac_airline_dest:
                        print(
                            "iLP data group for {}: (TAD) Type, Airline, Destination. AC_id: {}, Airline: {}, Airport:"
                            " {}.".format(ID, AC_id,
                                          kwargs["airline"],
                                          kwargs["DEST"]))
                        TAD_temp = self.TAD.groupby(["AC_id", "airline", "DEST"]).get_group(
                            (AC_id, kwargs["airline"], kwargs["DEST"])).reset_index(
                            drop=True)
                        return *self.TXXf(TAD_temp), SID

                    elif (AC_id, kwargs["airline"], country) in self.ac_airline_country:
                        print(
                            "iLP data group for {}: (TADL) Type, Airline, Country. AC_id: {}, Airline: {}, Country: {}."
                            .format(ID, AC_id, kwargs["airline"], country))
                        TADL_temp = self.TADLG.get_group((AC_id, kwargs["airline"], country)).reset_index(drop=True)
                        return *self.TXXf(TADL_temp), SID

                    elif (AC_id, kwargs["airline"], continent) in self.ac_airline_continent:
                        print(
                            "iLP data group for {}: (TADC) Type, Airline, Continent. AC_id: {}, Airline: {}, Continent:"
                            " {}.".format(ID, AC_id, kwargs["airline"], continent))
                        TADC_temp = self.TADCG.get_group((AC_id, kwargs["airline"], continent)).reset_index(drop=True)
                        return *self.TXXf(TADC_temp), SID

                if (AC_id, kwargs["DEST"]) in self.ac_dest:
                    print("iLP data group for {}: (TD) Type, No Airline, Destination. AC_id: {}, Airport: {}."
                          .format(ID, AC_id, kwargs["DEST"]))
                    TD_temp = self.TDG.get_group((AC_id, kwargs["DEST"])).reset_index(drop=True)
                    return *self.TXXf(TD_temp), SID

                elif (AC_id, country) in self.ac_country:
                    print(
                        "iLP data group for {}: (TCOU) Type, No Airline, Country. AC_id: {}, Country: {}."
                        .format(ID, AC_id, country))
                    TCOU_temp = self.TCOUG.get_group((AC_id, country)).reset_index(drop=True)
                    return *self.TXXf(TCOU_temp), SID

                elif (AC_id, continent) in self.ac_continent:
                    print(
                        "iLP data group for {}: (TCON) Type, No Airline, Continent. AC_id: {}, Continent: {}."
                        .format(ID, AC_id, continent))
                    TCON_temp = self.TCONG.get_group((AC_id, continent)).reset_index(drop=True)
                    return *self.TXXf(TCON_temp), SID

                swA = True
            if "DEST" not in kwargs or swA:
                if "airline" in kwargs:
                    if (AC_id, kwargs["airline"]) in self.ac_airline:
                        print(
                            "iLP data group for {}: (TA) Type, Airline, No Destination.  AC_id: {}, Airline: {}."
                            .format(ID, AC_id, kwargs["airline"]))
                        TA_temp = self.TAG.get_group((AC_id, kwargs["airline"])).reset_index(drop=True)
                        return *self.TXXf(TA_temp), SID

                if AC_id in self.AC_ids:
                    print(
                        "iLP data group for {}: (TT) Type, No Airline, No Destination. AC_id: {}.".format(ID, AC_id))
                    TT_temp = self.TTG.get_group((AC_id,)).reset_index(drop=True)
                    return *self.TXXf(TT_temp), SID

        # Check for only the AC_type
        if "DEST" in kwargs:
            if "airline" in kwargs:
                if (AC_type, kwargs["airline"], kwargs["DEST"]) in self.ac_airline_dest:
                    print(
                        "iLP data group for {}: (TAD) Type, Airline, Destination. AC_id: {}, Airline: {}, Airport:"
                        " {}.".format(ID, AC_type,
                                      kwargs["airline"],
                                      kwargs["DEST"]))
                    TAD_temp = self.TAD.groupby(["AC_id", "airline", "DEST"]).get_group(
                        (AC_type, kwargs["airline"], kwargs["DEST"])).reset_index(
                        drop=True)
                    return *self.TXXf(TAD_temp), SID

                elif (AC_type, kwargs["airline"], country) in self.ac_airline_country:
                    print(
                        "iLP data group for {}: (TADL) Type, Airline, Country. AC_type: {}, Airline: {}, Country: {}."
                        .format(ID, AC_type, kwargs["airline"], country))
                    TADL_temp = self.TADLG.get_group((AC_type, kwargs["airline"], country)).reset_index(drop=True)
                    return *self.TXXf(TADL_temp), SID

                elif (AC_type, kwargs["airline"], continent) in self.ac_airline_continent:
                    print(
                        "iLP data group for {}: (TADC) Type, Airline, Continent. AC_type: {}, Airline: {}, Continent:"
                        " {}.".format(ID, AC_type, kwargs["airline"], continent))
                    TADC_temp = self.TADCG.get_group((AC_type, kwargs["airline"], continent)).reset_index(
                        drop=True)
                    return *self.TXXf(TADC_temp), SID

            if (AC_type, kwargs["DEST"]) in self.ac_dest:
                print("iLP data group for {}: (TD) Type, No Airline, Destination. AC_type: {}, Airport: {}."
                      .format(ID, AC_type, kwargs["DEST"]))
                TD_temp = self.TDG.get_group((AC_type, kwargs["DEST"])).reset_index(drop=True)
                return *self.TXXf(TD_temp), SID

            elif (AC_type, country) in self.ac_country:
                print(
                    "iLP data group for {}: (TCOU) Type, No Airline, Country. AC_type: {}, Country: {}."
                    .format(ID, AC_type, country))
                TCOU_temp = self.TCOUG.get_group((AC_type, country)).reset_index(drop=True)
                return *self.TXXf(TCOU_temp), SID

            elif (AC_type, continent) in self.ac_continent:
                print(
                    "iLP data group for {}: (TCON) Type, No Airline, Continent. AC_type: {}, Continent: {}."
                    .format(ID, AC_type, continent))
                TCON_temp = self.TCONG.get_group((AC_type, continent)).reset_index(drop=True)
                return *self.TXXf(TCON_temp), SID

            swA = True
        if "DEST" not in kwargs or swA:
            if "airline" in kwargs:
                if (AC_type, kwargs["airline"]) in self.ac_airline:
                    print(
                        "iLP data group for {}: (TA) Type, Airline, No Destination.  AC_type: {}, Airline: {}."
                        .format(ID, AC_type, kwargs["airline"]))
                    TA_temp = self.TAG.get_group((AC_type, kwargs["airline"])).reset_index(drop=True)
                    return *self.TXXf(TA_temp), SID

            if AC_type in self.AC_ids:
                print(
                    "iLP data group for {}: (TT) Type, No Airline, No Destination. AC_type: {}.".format(ID,
                                                                                                        AC_type))
                TT_temp = self.TTG.get_group(AC_type).reset_index(drop=True)
                return *self.TXXf(TT_temp), SID

        print("iLP data group for {}: No data available. AC_id: {}. Using B744 performance.".format(ID, AC_id))
        TT_temp = self.TTG.get_group("B744").reset_index(drop=True)

        return *self.TXXf(TT_temp), SID

    def TXXf(self, df):
        """
        Function: Returns the interpolated dataframe for all altitudes and maximum speed slope;

        Args:
            df:         Relevant dataframe for the required degree of accuracy

        Returns:
            final_df:     Dataframe with IAS/CAS and ROC for aircraft agent
            IAS_slope_max: maximum slope of IAS per flight level from the dataframe

        Created by: Arham Elahi
        Date: July 2024

        Edited by: Albert Chou
        Date: May 2025
        """
        FL_temp = np.arange(0, df["ALT"].max() + 1, 1)
        df_IAS = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "IAS"].values, kind=self.kind,
                             fill_value="extrapolate")(FL_temp)
        df_MACH = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "MACH"].values, kind=self.kind,
                              fill_value="extrapolate")(FL_temp)
        df_COLalt = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "COLalt"].values, kind=self.kind,
                                fill_value="extrapolate")(FL_temp)
        df_ROC = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "ROCD"].values, kind=self.kind,
                             fill_value="extrapolate")(FL_temp)
        df_IASm = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "IAS_m"].values, kind=self.kind,
                              fill_value="extrapolate")(FL_temp)
        df_IASsd = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "IAS_sd"].values, kind=self.kind,
                               fill_value="extrapolate")(FL_temp)
        df_MACHm = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "MACH_m"].values, kind=self.kind,
                               fill_value="extrapolate")(FL_temp)
        df_MACHsd = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "MACH_sd"].values, kind=self.kind,
                                fill_value="extrapolate")(FL_temp)
        df_ROCDm = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "ROCD_m"].values, kind=self.kind,
                               fill_value="extrapolate")(FL_temp)
        df_ROCDsd = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "ROCD_sd"].values, kind=self.kind,
                                fill_value="extrapolate")(FL_temp)
        IAS_slope_max = df["IAS"].diff().max() / (df.at[1, "ALT"] - df.at[0, "ALT"])

        df_min_calt = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "min_calt"].values, kind=self.kind,
                                  fill_value="extrapolate")(FL_temp)

        df_max_calt = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "max_calt"].values, kind=self.kind,
                                  fill_value="extrapolate")(FL_temp)

        df_mode_calt = sc.interp1d(df.loc[:, "ALT"].values, df.loc[:, "mode_calt"].values, kind=self.kind,
                                   fill_value="extrapolate")(FL_temp)

        data = {"ALT": FL_temp, "IAS": df_IAS, "ROCD": df_ROC, "MACH": df_MACH, "COLalt": df_COLalt, "IAS_m": df_IASm,
                "IAS_sd": df_IASsd, "MACH_m": df_MACHm, "MACH_sd": df_MACHsd, "ROCD_m": df_ROCDm, "ROCD_sd": df_ROCDsd,
                "min_calt": df_min_calt, "max_calt": df_max_calt, "mode_calt": df_mode_calt}

        final_df = pd.DataFrame(data=data)

        return final_df, IAS_slope_max

    def IAS_ROC(self, dfs, alts, spds, sds_spd=None, sds_roc=None, calts=None):
        """
        Function: Return estimated speed (IAS/CAS), ROC, and MACH for all aircraft agents with take-off switch set to True.

        Args:
            dfs:        Array with dataframes of all agents
            alts:       Array with the flight altitude of all agents
            spds:       Array with the flight speeds of all agents
            sds_spd:    Array with performance levels for speed of all agents, -ve means slower speed
            sds_roc:    Array with performance levels for rate of climb of all agents, -ve means slower climb
        Returns:
            return_1: IAS/CAS array with all estimated aircraft speeds
            return_2: ROC array with all estimated aircraft rate of climb
            return_3: MACH array with all estimated aircraft mach numbers
            return_4: COLalt array with estimated crossover level altitude

        Created by: Lars Dijkstra
        Date: Somewhere early/mid-October 2022

        Edited by: Amre Dardor
        Date: July 2023

        Edited by: Arham Elahi
        Date: July 2024
        """
        if sds_spd is None:
            sds_spd = np.zeros(np.size(alts))
        if sds_roc is None:
            sds_roc = np.zeros(np.size(alts))
        if calts is None:
            calts = np.zeros(np.size(alts))
        alts = alts / self.ft/ 100  # Conversion to Flight Level
        spds = spds / self.kts  # Conversion to knots
        calts = calts / self.ft / 100  # Conversion to Flight Level
        array = np.array(
            [self.values(dfs[i], alts[i], spds[i], sds_spd[i], sds_roc[i], calts[i]) for i in range(len(alts))])

        return np.array([element[0] * self.kts for element in array]), np.array(
            [element[1] * self.fpm for element in array]), np.array([element[2] for element in array]), np.array(
            [element[3] for element in array])

    def values(self, df, alt, spd, sd_spd, sd_roc, calt):
        """
        Function: Return estimated speed (IAS/CAS) and ROC for one aircraft agent with take-off switch set to True. For
                  agents with a take-off switch set to False, the program returns (False, False) (reducing computations).
                  Moreover, it also prohibits aircraft from climbing higher than their maximum altitude available in
                  the specific data-set.

        Args:
            df:         Array with dataframes of one agent
            alt:        Array with the flight altitude of one agent
            spd:        Array with the indicated airspeed of one agent
            sd_spd:     Array with speed performance level of one agent
            sd_roc:     Array with climb performance level of one agent
            calt:       Array with the cruise altitude of one agent
        Returns:
            return_1: IAS/CAS array with all estimated aircraft speeds
            return_2: ROC array with all estimated aircraft rate of climb
            return_3: MACH array with all estimated aircraft Mach numbers
            return_4: COLalt array with estimated crossover level altitude

        Created by: Lars Dijkstra
        Date: Somewhere early/mid-October 2022

        Edited by: Amre Dardor
        Date: July 2023

        Edited by: Arham Elahi
        Date: July 2024
        """
        # Check whether we should return a value, by checking whether there is a dataframe available (an aircraft agent
        # only receives a dataframe when it has received a take-off switch, else 'False' is assigned).

        if isinstance(df, bool):
            return False, False, False, False

        if alt < 0:
            alt = 0

        # Obtain row in dataframe corresponding to the altitude input.
        temp = df.loc[df['ALT'] == int(alt)].iloc[:1]

        # If row not found, return (False, False)
        if temp.empty:
            return False, False, False, False

        if not sd_spd.any():
            # Define target Speed, Mach, Crossover Altitude and Rate of Climb
            tar_IAS = temp.iloc[0]["IAS"]
            tar_MACH = temp.iloc[0]["MACH"]
        else:
            # Add/subtract the performance level (in standard deviations) from the mean value
            tar_IAS = temp.iloc[0]["IAS_m"] + sd_spd * temp.iloc[0]["IAS_sd"]
            tar_MACH = temp.iloc[0]["MACH_m"] + sd_spd * temp.iloc[0]["MACH_sd"]

        if not sd_roc.any():
            tar_ROCD = temp.iloc[0]['ROCD']
        else:
            tar_ROCD = temp.iloc[0]['ROCD_m'] + sd_roc * temp.iloc[0]["ROCD_sd"]

        tar_COLalt = temp.iloc[0]["COLalt"]

        # Apply minimum ROCD for the climb profile
        if alt < 1 and spd > 0.95 * tar_IAS:
            tar_ROCD = max(tar_ROCD, 2000)
        elif alt >= calt or alt < 1:
            tar_ROCD = 0
        else:
            tar_ROCD = max(tar_ROCD, 500)

        # An aircraft may not climb higher than its max altitude from the data-set, therefore return a ROC of 0. Else,
        # return the corresponding ROC, the ROC should always be positive otherwise airplane will get stuck at the same
        # altitude.
        return tar_IAS, tar_ROCD, tar_MACH, tar_COLalt