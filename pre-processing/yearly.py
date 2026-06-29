import xarray as xr


def yearly_sum(scenario: int, year: int, def_option: str = "Raw"):
    """Berechnet die jährliche Summe der Wasserflüsse (Netzwerk, Niederschlag, Verdunstung) 
    und speichert das Ergebnis in einer neuen NetCDF-Datei.

    Args:
        scenario (int): SSP Szenario-Nummer.
        year (int): Jahr des Szenarios.
        base_path (str, optional): Verzeichnis, in dem die monatlichen Rohdaten liegen.
    """
    ds_yearly = None

    for month in range(1, 13):
        if def_option == "Raw":
            nc_filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
        elif def_option == "BaU":
            nc_filepath = f"./pre-processed-data/water_BaU/scenario_ssp{scenario}_decade{year}_month{month:02d}_BaU.nc"
        elif def_option == "Gov":
            nc_filepath = f"./pre-processed-data/water_Gov/scenario_ssp{scenario}_decade{year}_month{month:02d}_Gov.nc"
        elif def_option == "Sec":
            nc_filepath = f"./pre-processed-data/water_Sec/scenario_ssp{scenario}_decade{year}_month{month:02d}_Sec.nc"

        # Datensatz für den aktuellen Monat laden
        ds_month = xr.open_dataset(nc_filepath)

        if ds_yearly is None:
            # Beim ersten Monat (Januar): Eine vollständige Kopie als Basis erstellen.
            ds_yearly = ds_month.copy(deep=True)
        else:
            # Für alle weiteren Monate: Die Werte der drei Variablen einfach aufaddieren.
            # xarray kümmert sich automatisch darum, dass die Matrizen korrekt addiert werden.
            ds_yearly["network"] += ds_month["network"]
            ds_yearly["prec"] += ds_month["prec"]
            ds_yearly["evap"] += ds_month["evap"]

        # Den monatlichen Datensatz wieder schließen, um RAM freizugeben und File-Locks zu vermeiden
        ds_month.close()

    if def_option == "Raw":
        output_filepath = f"./pre-processed-data/water_Raw/scenario_ssp{scenario}_decade{year}_Raw.nc"
    elif def_option == "BaU":
        output_filepath = f"./pre-processed-data/water_BaU/scenario_ssp{scenario}_decade{year}_BaU.nc"
    elif def_option == "Gov":
        output_filepath = f"./pre-processed-data/water_Gov/scenario_ssp{scenario}_decade{year}_Gov.nc"
    elif def_option == "Sec":
        output_filepath = f"./pre-processed-data/water_Sec/scenario_ssp{scenario}_decade{year}_Sec.nc"

    # Das aufaddierte Dataset als neue NetCDF-Datei speichern
    ds_yearly.to_netcdf(output_filepath)
    ds_yearly.close()

    print(f"Jährliche Summe erfolgreich berechnet und gespeichert unter: {output_filepath}")


def execute_yearly_sum():
    scenarios = [245, 370, 585]  # Beispielhafte Szenarien
    years_raw = range(2030, 2100)
    years_def = range(2030, 2051)

    for scenario in scenarios:
        for year in years_raw:
            print(f"Verarbeite Szenario {scenario}, Jahr {year} (Raw)")
            yearly_sum(scenario, year)

            print(f"Verarbeite Szenario {scenario}, Jahr {year} (Sec)")
            yearly_sum(scenario, year, def_option="Sec")

        for year in years_def:
            print(f"Verarbeite Szenario {scenario}, Jahr {year} (BaU)")
            yearly_sum(scenario, year, def_option="BaU")

            print(f"Verarbeite Szenario {scenario}, Jahr {year} (Gov)")
            yearly_sum(scenario, year, def_option="Gov")


if __name__ == "__main__":
    execute_yearly_sum()
